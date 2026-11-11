//! `breccia-server` — an OpenAI-compatible HTTP surface over one generation session.
//!
//! With one sequence slot ([`Backend::slots`] = 1) the server is a single-threaded
//! HTTP/1.1 loop: accept, read one request, generate, respond. With more, connection
//! threads hand requests to the engine thread, which decodes every admitted request
//! in one batched step per token ([`serve`]). The crate stays dependency-free apart
//! from `serde_json`.
//!
//! Routes:
//!   `GET  /health`              liveness + loaded model
//!   `POST /v1/completions`      `{prompt, max_tokens}` → text completion
//!   `POST /v1/chat/completions` `{messages, max_tokens}` → ChatML completion
//!   `GET  /v1/models`           the served model name
//!
//! The caller supplies a [`Backend`]: tokenization and generation live in the CLI
//! (which owns the model, session and tokenizer), so this crate never links the
//! engine.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use serde_json::{json, Value};

/// What the server needs from the engine.
pub trait Backend {
    /// Model name for `/v1/models` and the response `model` field.
    fn model_name(&self) -> String;
    /// Text → token ids (the model's own tokenizer).
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String>;
    /// Ids → text.
    fn decode_text(&self, ids: &[u32]) -> String;
    /// Generate at most `max_tokens` tokens after `prompt`; returns the new ids.
    fn complete(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>, String>;
    /// The end-of-sequence id, so the server can trim it and report `stop`.
    fn eos(&self) -> Option<u32>;
    /// A control token's id by text (e.g. `<|im_start|>`), for the chat template.
    fn special(&self, text: &str) -> Option<u32> {
        let _ = text;
        None
    }
    /// Encode the request's image number `index` (the raw file bytes) for the next
    /// `complete`; returns how many `<|image_pad|>` cells it fills. The prompt then
    /// holds that many pads per image, in order. Index 0 drops earlier ones.
    fn set_image(&mut self, index: usize, data: &[u8]) -> Result<usize, String> {
        let _ = (index, data);
        Err("this server takes no image input (start it with --mmproj)".into())
    }
    /// Drop encoded images that no `complete` will use (a request failed).
    fn clear_images(&mut self) {}
    /// Speculative-decoding hint for the next `complete`: `explicit` is the request's
    /// `"mtp"` field, `code` whether its last user turn asks for code ([`wants_code`]).
    /// Returns whether that `complete` will spec-decode (it then runs alone).
    fn spec_hint(&mut self, explicit: Option<bool>, code: bool) -> bool {
        let _ = (explicit, code);
        false
    }
    /// Sequence slots that decode together (1 = one request at a time via `complete`).
    fn slots(&self) -> usize {
        1
    }
    /// Start a text request on free slot `slot`: prefill `prompt` (plain decode) and
    /// return `(first token, tokens allowed)`, or `None` when it does not fit beside
    /// the running slots yet (it is retried after one finishes).
    fn admit(&mut self, slot: usize, prompt: &[u32], max_tokens: usize) -> Result<Option<(u32, usize)>, String> {
        let _ = (slot, prompt, max_tokens);
        Err("this backend has no sequence slots".into())
    }
    /// [`Self::admit`] that may hand back `(None, allowed)`: the prompt pass then runs one
    /// chunk per [`Self::feed`] call, between the live slots' decode steps, so they keep
    /// streaming while a long prompt goes in. Default: the whole pass in `admit`.
    fn admit_start(&mut self, slot: usize, prompt: &[u32], max_tokens: usize) -> Result<Option<(Option<u32>, usize)>, String> {
        Ok(self.admit(slot, prompt, max_tokens)?.map(|(t, n)| (Some(t), n)))
    }
    /// The next prompt chunk of slot `slot` (after `admit_start` gave `None`); the first
    /// generated token once the prompt is in. `decoding` = how many other slots are
    /// decoding (a backend may run a smaller chunk then).
    fn feed(&mut self, slot: usize, decoding: usize) -> Result<Option<u32>, String> {
        let _ = (slot, decoding);
        Err("this backend has no chunked admission".into())
    }
    /// [`Self::feed`] for every slot still feeding (oldest first): a backend may put
    /// several prompts' chunks through one pass. One result per slot. Default: only
    /// the oldest slot gets a chunk.
    fn feed_many(&mut self, slots: &[usize], decoding: usize) -> Vec<Result<Option<u32>, String>> {
        let mut r = vec![self.feed(slots[0], decoding)];
        r.resize_with(slots.len(), || Ok(None));
        r
    }
    /// One decode step: feed `rows[i] = (slot, token)`, return each row's next token.
    fn step(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        let _ = rows;
        Err("this backend has no sequence slots".into())
    }
    /// Run a request alone on free slot `slot` (spec decode or images, nothing else live).
    /// `waiting` turns true when another request arrives: a backend may then stop at a
    /// token boundary and say `yielded`; the slot holds the prompt and every id but the
    /// last, and continues in [`Self::step`]. Returns `(ids, tokens allowed, yielded)`;
    /// the server releases a slot that did not yield.
    fn complete_alone(&mut self, slot: usize, prompt: &[u32], max_tokens: usize, waiting: &Arc<AtomicBool>) -> Result<(Vec<u32>, usize, bool), String> {
        let _ = (slot, waiting);
        Ok((self.complete(prompt, max_tokens)?, max_tokens, false))
    }
    /// Slot `slot`'s request is done: free what it holds.
    fn release(&mut self, slot: usize) {
        let _ = slot;
    }
    /// Prompt + generation cells the backend holds, if bounded.
    fn context(&self) -> Option<usize> {
        None
    }
}

pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Response {
    fn json(status: u16, v: Value) -> Self {
        Self {
            status,
            body: serde_json::to_vec(&v).unwrap_or_else(|_| b"{}".to_vec()),
        }
    }
}

/// Serve until the listener errors. `addr` is `host:port`.
pub fn serve<B: Backend>(addr: &str, mut backend: B) -> Result<(), String> {
    let listener = TcpListener::bind(addr).map_err(|e| format!("binding {addr}: {e}"))?;
    eprintln!(
        "breccia-server: listening on http://{addr} (model {}, {} slot(s))",
        backend.model_name(),
        backend.slots()
    );
    if backend.slots() > 1 {
        return serve_slots(listener, backend);
    }
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                if let Err(e) = handle(stream, &mut backend) {
                    eprintln!("breccia-server: {e}");
                }
            }
            Err(e) => eprintln!("breccia-server: accept: {e}"),
        }
    }
    Ok(())
}

fn handle<B: Backend>(mut stream: TcpStream, backend: &mut B) -> Result<(), String> {
    let Some((method, path, body)) = read_request(&mut stream)? else {
        return Ok(());
    };
    let resp = dispatch(&method, &path, &body, backend);
    write_response(&mut stream, &resp)
}

/// One HTTP/1.1 request: `(method, path, body)`, or `None` on an empty connection.
fn read_request(stream: &mut TcpStream) -> Result<Option<(String, String, Vec<u8>)>, String> {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .ok();
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);

    let mut line = String::new();
    if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut len = 0usize;
    let mut expect_continue = false;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        if lower.starts_with("expect:") && lower.contains("100-continue") {
            expect_continue = true;
        }
    }
    if expect_continue {
        stream
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .map_err(|e| e.to_string())?;
    }
    if len > 64 << 20 {
        return Err(format!("request body of {len} bytes is over the 64 MiB cap"));
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    Ok(Some((method, path, body)))
}

fn write_response(stream: &mut TcpStream, resp: &Response) -> Result<(), String> {
    let reason = match resp.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        resp.status,
        resp.body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(&resp.body))
        .and_then(|_| stream.flush())
        .map_err(|e| e.to_string())
}

/// A request a connection thread read, for the engine thread to answer.
struct Job {
    method: String,
    path: String,
    body: Vec<u8>,
    reply: mpsc::Sender<Response>,
}

/// A completion request waiting for a slot (prepared once, on its first try).
struct Pending {
    reply: mpsc::Sender<Response>,
    chat: bool,
    req: Value,
    image: bool,
    p: Option<Prepared>,
}

/// A completion request decoding in a slot.
struct Live {
    reply: mpsc::Sender<Response>,
    p: Prepared,
    ids: Vec<u32>,
    allowed: usize,
    t0: std::time::Instant,
    /// When the last token came, and the worst wait between two (the stall a long admit causes).
    last: std::time::Instant,
    gap: f64,
    /// When the first token came (prompt pass done).
    first: std::time::Instant,
}

/// Several sequence slots: connection threads read requests and answer `/health` and
/// `/v1/models` themselves; the engine stays on this thread, admits completions into
/// free slots and decodes every live slot in one step per token. A request that
/// spec-decodes or carries images runs alone through `complete_alone`, when no slot is
/// live and nothing else is queued; a spec run hands over to batched decode as soon as
/// another request arrives (`waiting`). shortcut: an image request still runs to its end.
fn serve_slots<B: Backend>(listener: TcpListener, mut b: B) -> Result<(), String> {
    let (tx, rx) = mpsc::channel::<Job>();
    let name = b.model_name();
    let waiting = Arc::new(AtomicBool::new(false));
    let arrived = waiting.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    let (tx, name, arrived) = (tx.clone(), name.clone(), arrived.clone());
                    std::thread::spawn(move || {
                        if let Err(e) = handle_conn(stream, &name, &tx, &arrived) {
                            eprintln!("breccia-server: {e}");
                        }
                    });
                }
                Err(e) => eprintln!("breccia-server: accept: {e}"),
            }
        }
    });
    let ctx = b.context().unwrap_or(usize::MAX);
    let mut queue: VecDeque<Pending> = VecDeque::new();
    let mut live: Vec<Option<Live>> = (0..b.slots()).map(|_| None).collect();
    // `BRECCIA_SERVE_DECODE_SHARE=0.5`: while prompts feed and others decode, decode gets this
    // share of the wall (feed passes wait until the steps since have paid it); unset = one pass per step
    let share = std::env::var("BRECCIA_SERVE_DECODE_SHARE").ok().and_then(|v| v.parse::<f64>().ok()).filter(|s| *s > 0.0 && *s < 1.0);
    let (mut feed_s, mut step_s) = (0.0f64, 0.0f64);
    loop {
        let busy = live.iter().any(Option::is_some);
        let mut jobs: Vec<Job> = Vec::new();
        if !busy && queue.is_empty() {
            jobs.push(rx.recv().map_err(|_| "the listener thread is gone".to_string())?);
        }
        jobs.extend(rx.try_iter());
        for job in jobs {
            if let Some(p) = intake(job, &mut b) {
                queue.push_back(p);
            }
        }
        // admit in arrival order; what cannot start yet keeps its place
        let mut wait = VecDeque::new();
        while let Some(mut q) = queue.pop_front() {
            let busy = live.iter().any(Option::is_some);
            let Some(slot) = live.iter().position(Option::is_none) else {
                wait.push_back(q);
                continue;
            };
            if q.image && busy {
                wait.push_back(q);
                continue;
            }
            let p = match q.p.take() {
                Some(p) => p,
                None => match prepare(&q.req, &mut b, q.chat) {
                    Ok(p) => p,
                    Err(r) => {
                        let _ = q.reply.send(r);
                        continue;
                    }
                },
            };
            let spec = b.spec_hint(p.mtp, p.code);
            if !busy && (q.image || spec) {
                // clear, then drain: a request sent after this load sets `waiting` again
                waiting.store(false, Ordering::SeqCst);
                for job in rx.try_iter() {
                    if let Some(p) = intake(job, &mut b) {
                        queue.push_back(p);
                    }
                }
            }
            // spec only with nothing else waiting; otherwise it joins the batch (plain decode)
            if !busy && (q.image || (spec && queue.is_empty())) {
                let t0 = std::time::Instant::now();
                match b.complete_alone(slot, &p.prompt, p.max_tokens, &waiting) {
                    Ok((ids, allowed, true)) => {
                        let now = std::time::Instant::now();
                        eprintln!("breccia-server: slot {slot}: spec run handed over after {} tokens ({:.2} s)", ids.len(), (now - t0).as_secs_f64());
                        live[slot] = Some(Live { reply: q.reply, p, ids, allowed, t0, last: now, gap: 0.0, first: now });
                    }
                    Ok((ids, _, false)) => {
                        b.release(slot);
                        let _ = q.reply.send(finish(&mut b, &p, ids, t0.elapsed().as_secs_f64()));
                    }
                    Err(e) => {
                        b.release(slot);
                        eprintln!("breccia-server: request failed: {e}");
                        let _ = q.reply.send(Response::json(500, json!({"error": {"message": e}})));
                    }
                }
                continue;
            }
            match b.admit_start(slot, &p.prompt, p.max_tokens) {
                // a slot still feeding its prompt holds no ids: it joins the steps once fed
                Ok(Some((tok, allowed))) => {
                    live[slot] = Some(Live { reply: q.reply, p, ids: tok.into_iter().collect(), allowed, t0: std::time::Instant::now(), last: std::time::Instant::now(), gap: 0.0, first: std::time::Instant::now() });
                }
                Ok(None) => {
                    q.p = Some(p);
                    wait.push_back(q);
                }
                Err(e) => {
                    b.release(slot);
                    eprintln!("breccia-server: request failed: {e}");
                    let _ = q.reply.send(Response::json(500, json!({"error": {"message": e}})));
                }
            }
        }
        queue = wait;
        retire(&mut live, &mut b, ctx);
        // one prompt pass (oldest admission first) per decode step
        let mut feeding: Vec<_> = live
            .iter()
            .enumerate()
            .filter_map(|(s, l)| l.as_ref().filter(|l| l.ids.is_empty()).map(|l| (l.t0, s)))
            .collect();
        feeding.sort();
        let feeding: Vec<usize> = feeding.into_iter().map(|x| x.1).collect();
        let decoding = live.iter().filter(|l| l.as_ref().is_some_and(|l| !l.ids.is_empty())).count();
        if feeding.is_empty() || decoding == 0 {
            (feed_s, step_s) = (0.0, 0.0);
        }
        let shared = share.is_some() && !feeding.is_empty() && decoding > 0;
        let feeding = if shared && !feed_now(share, feed_s, step_s) { Vec::new() } else { feeding };
        let t = std::time::Instant::now();
        let fed = if feeding.is_empty() { Vec::new() } else { b.feed_many(&feeding, decoding) };
        if shared {
            feed_s += t.elapsed().as_secs_f64();
        }
        for (s, r) in feeding.into_iter().zip(fed) {
            match r {
                Ok(Some(tok)) => {
                    let l = live[s].as_mut().expect("a live slot");
                    l.ids.push(tok);
                    l.last = std::time::Instant::now();
                    l.first = l.last;
                }
                Ok(None) => {}
                Err(e) => {
                    b.release(s);
                    eprintln!("breccia-server: request failed: {e}");
                    let l = live[s].take().expect("a live slot");
                    let _ = l.reply.send(Response::json(500, json!({"error": {"message": e}})));
                }
            }
        }
        let rows: Vec<(usize, u32)> = live
            .iter()
            .enumerate()
            .filter_map(|(s, l)| Some((s, *l.as_ref()?.ids.last()?)))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let t = std::time::Instant::now();
        match b.step(&rows) {
            Ok(next) => {
                let now = std::time::Instant::now();
                if shared {
                    step_s += (now - t).as_secs_f64();
                }
                for (&(s, _), t) in rows.iter().zip(next) {
                    let l = live[s].as_mut().expect("a live slot");
                    l.ids.push(t);
                    l.gap = l.gap.max((now - l.last).as_secs_f64());
                    l.last = now;
                }
                retire(&mut live, &mut b, ctx);
            }
            Err(e) => {
                eprintln!("breccia-server: batched step failed: {e}");
                for (s, l) in live.iter_mut().enumerate() {
                    if let Some(l) = l.take() {
                        b.release(s);
                        let _ = l.reply.send(Response::json(500, json!({"error": {"message": e.clone()}})));
                    }
                }
            }
        }
    }
}

/// Decode share: feed again once decode's time since the last reset reaches `share` of the total.
fn feed_now(share: Option<f64>, feed_s: f64, step_s: f64) -> bool {
    share.map_or(true, |s| step_s >= feed_s * s / (1.0 - s))
}

#[test]
fn feed_now_pays_the_decode_share() {
    assert!(feed_now(None, 9.0, 0.0));
    assert!(feed_now(Some(0.5), 0.0, 0.0));
    assert!(!feed_now(Some(0.5), 8.0, 7.9));
    assert!(feed_now(Some(0.5), 8.0, 8.0));
    assert!(!feed_now(Some(0.75), 2.0, 5.9));
    assert!(feed_now(Some(0.75), 2.0, 6.0));
}

/// Answer and free the slots that are done: max tokens, EOS, or the context's end.
fn retire<B: Backend>(live: &mut [Option<Live>], b: &mut B, ctx: usize) {
    // `BRECCIA_SERVE_IGNORE_EOS=1`: benches only, every request runs to max_tokens
    let eos = b.eos().filter(|_| std::env::var("BRECCIA_SERVE_IGNORE_EOS").map_or(true, |v| v != "1"));
    for (s, slot) in live.iter_mut().enumerate() {
        let done = slot.as_ref().is_some_and(|l| {
            l.ids.len() >= l.allowed
                || eos.is_some_and(|e| l.ids.last() == Some(&e))
                || l.p.prompt.len() + l.ids.len() >= ctx
        });
        if let Some(l) = slot.take_if(|_| done) {
            b.release(s);
            let secs = l.t0.elapsed().as_secs_f64();
            let ttft = (l.first - l.t0).as_secs_f64();
            let dec = l.ids.len().saturating_sub(1) as f64 / l.first.elapsed().as_secs_f64().max(1e-9);
            eprintln!("breccia-server: slot {s} done: {} tokens in {secs:.2} s, worst token gap {:.0} ms, first token {ttft:.2} s, decode {dec:.1} tok/s", l.ids.len(), l.gap * 1e3);
            let _ = l.reply.send(finish(b, &l.p, l.ids, secs));
        }
    }
}

/// The engine thread's first look at a request: non-completion routes are answered
/// here, a completion becomes [`Pending`] (or a 400 for bad JSON).
fn intake<B: Backend>(job: Job, b: &mut B) -> Option<Pending> {
    let path = job.path.split('?').next().unwrap_or(&job.path);
    let chat = match (job.method.as_str(), path) {
        ("POST", "/v1/chat/completions") => true,
        ("POST", "/v1/completions") => false,
        _ => {
            let _ = job.reply.send(dispatch(&job.method, &job.path, &job.body, b));
            return None;
        }
    };
    match serde_json::from_slice::<Value>(&job.body) {
        Ok(req) => {
            let image = req.get("messages").and_then(|m| m.as_array()).is_some_and(|msgs| {
                msgs.iter().any(|m| {
                    m.get("content").and_then(|c| c.as_array()).is_some_and(|parts| {
                        parts.iter().any(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url"))
                    })
                })
            });
            Some(Pending { reply: job.reply, chat, req, image, p: None })
        }
        Err(e) => {
            let _ = job.reply.send(Response::json(400, json!({"error": {"message": format!("invalid JSON: {e}")}})));
            None
        }
    }
}

fn handle_conn(mut stream: TcpStream, name: &str, tx: &mpsc::Sender<Job>, arrived: &AtomicBool) -> Result<(), String> {
    let Some((method, path, body)) = read_request(&mut stream)? else {
        return Ok(());
    };
    let resp = match info_route(&method, path.split('?').next().unwrap_or(&path), name) {
        Some(r) => r,
        None => {
            let (reply, done) = mpsc::channel();
            if method == "POST" && path.contains("completions") {
                arrived.store(true, Ordering::SeqCst); // a running spec request hands over
            }
            tx.send(Job { method, path, body, reply }).map_err(|_| "the engine thread is gone")?;
            done.recv().map_err(|_| "the engine dropped the request")?
        }
    };
    write_response(&mut stream, &resp)
}

/// `/health` and `/v1/models`: answered without the engine.
fn info_route(method: &str, path: &str, name: &str) -> Option<Response> {
    match (method, path) {
        ("GET", "/health") => Some(Response::json(
            200,
            json!({"status": "ok", "model": name, "backend": "breccia"}),
        )),
        ("GET", "/v1/models") => Some(Response::json(
            200,
            json!({"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"}]}),
        )),
        _ => None,
    }
}

/// Route one request (exposed so the HTTP surface is testable without a model).
pub fn dispatch<B: Backend>(method: &str, path: &str, body: &[u8], b: &mut B) -> Response {
    let path = path.split('?').next().unwrap_or(path);
    if let Some(r) = info_route(method, path, &b.model_name()) {
        return r;
    }
    match (method, path) {
        ("POST", "/v1/completions") => completions(body, b, false),
        ("POST", "/v1/chat/completions") => completions(body, b, true),
        (_, "/v1/completions") | (_, "/v1/chat/completions") => {
            Response::json(405, json!({"error": {"message": "use POST"}}))
        }
        _ => Response::json(
            404,
            json!({"error": {"message": format!("no route {path}")}}),
        ),
    }
}

fn completions<B: Backend>(body: &[u8], b: &mut B, chat: bool) -> Response {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return Response::json(
                400,
                json!({"error": {"message": format!("invalid JSON: {e}")}}),
            )
        }
    };
    let p = match prepare(&req, b, chat) {
        Ok(p) => p,
        Err(r) => return r,
    };
    b.spec_hint(p.mtp, p.code);
    run_alone(b, &p)
}

/// A completion request, tokenized (its images set on the backend) and ready to run.
struct Prepared {
    chat: bool,
    prompt: Vec<u32>,
    max_tokens: usize,
    mtp: Option<bool>,
    code: bool,
}

fn prepare<B: Backend>(req: &Value, b: &mut B, chat: bool) -> Result<Prepared, Response> {
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;

    let prompt = if chat {
        match chat_prompt(req, b).and_then(|(ids, imgs)| with_images(ids, imgs, b)) {
            Ok(ids) => ids,
            Err(e) => return Err(Response::json(400, json!({"error": {"message": e}}))),
        }
    } else {
        b.clear_images(); // images only ever come with the chat request that sent them
        match req.get("prompt") {
            Some(Value::String(s)) => match b.tokenize(s) {
                Ok(ids) => ids,
                Err(e) => return Err(Response::json(500, json!({"error": {"message": e}}))),
            },
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect(),
            _ => {
                return Err(Response::json(
                    400,
                    json!({"error": {"message": "prompt must be a string or an id array"}}),
                ))
            }
        }
    };
    if prompt.is_empty() {
        return Err(Response::json(400, json!({"error": {"message": "empty prompt"}})));
    }
    let code = if chat {
        last_user_text(req).is_some_and(|t| wants_code(&t))
    } else {
        req.get("prompt").and_then(|v| v.as_str()).is_some_and(wants_code)
    };
    Ok(Prepared { chat, prompt, max_tokens, mtp: req.get("mtp").and_then(|v| v.as_bool()), code })
}

/// Run one prepared request on its own through [`Backend::complete`].
fn run_alone<B: Backend>(b: &mut B, p: &Prepared) -> Response {
    let t0 = std::time::Instant::now();
    match b.complete(&p.prompt, p.max_tokens) {
        Ok(ids) => finish(b, p, ids, t0.elapsed().as_secs_f64()),
        Err(e) => {
            eprintln!("breccia-server: request failed: {e}");
            Response::json(500, json!({"error": {"message": e}}))
        }
    }
}

/// The OpenAI-shaped reply for generated `ids` (cut at EOS).
fn finish<B: Backend>(b: &mut B, p: &Prepared, ids: Vec<u32>, seconds: f64) -> Response {
    let (chat, prompt_ids) = (p.chat, &p.prompt);
    let (ids, finish_reason) = match b.eos().and_then(|eos| ids.iter().position(|&t| t == eos)) {
        Some(i) => (ids[..i].to_vec(), "stop"),
        None => (ids, "length"),
    };
    let text = b.decode_text(&ids);
    let model = b.model_name();
    let usage = json!({
        "prompt_tokens": prompt_ids.len(),
        "completion_tokens": ids.len(),
        "total_tokens": prompt_ids.len() + ids.len(),
    });
    let secs = if seconds > 0.0 { seconds } else { 1e-9 };
    if chat {
        Response::json(
            200,
            json!({
                "id": format!("chatcmpl-{}", now_id()),
                "object": "chat.completion",
                "created": now_secs(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }],
                "usage": usage,
                "breccia": {"seconds": seconds, "tokens_per_s": ids.len() as f64 / secs},
            }),
        )
    } else {
        Response::json(
            200,
            json!({
                "id": format!("cmpl-{}", now_id()),
                "object": "text_completion",
                "created": now_secs(),
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
                "usage": usage,
                "breccia": {"seconds": seconds, "tokens_per_s": ids.len() as f64 / secs},
            }),
        )
    }
}

/// Encode the request's images one at a time (in prompt order) and widen each
/// one's `<|image_pad|>` to its cell count. Stops at the first image that would
/// overflow the context, before encoding the rest.
fn with_images<B: Backend>(ids: Vec<u32>, images: Vec<Vec<u8>>, b: &mut B) -> Result<Vec<u32>, String> {
    if images.is_empty() {
        return Ok(ids);
    }
    let pad = b.special("<|image_pad|>").ok_or("the model has no <|image_pad|> token")?;
    if ids.iter().filter(|&&t| t == pad).count() != images.len() {
        return Err("the text holds a literal <|image_pad|>".into());
    }
    let mut total = ids.len() - images.len(); // text tokens
    let mut cells = Vec::with_capacity(images.len());
    for (i, data) in images.iter().enumerate() {
        let n = b.set_image(i, data);
        total += n.as_ref().map_or(0, |&n| n);
        let err = match (n, b.context()) {
            (Err(e), _) => Some(e),
            (Ok(_), Some(ctx)) if total >= ctx => Some(format!(
                "image {} of {} brings the prompt to {total} tokens, over the {ctx}-token context",
                i + 1,
                images.len()
            )),
            (Ok(n), _) => {
                cells.push(n);
                None
            }
        };
        if let Some(e) = err {
            // no complete() follows: don't leave this request's images for the next one
            b.clear_images();
            return Err(e);
        }
    }
    let mut out = Vec::with_capacity(total);
    let mut cells = cells.into_iter();
    for t in ids {
        match (t == pad).then(|| cells.next()).flatten() {
            Some(n) => out.extend(std::iter::repeat_n(pad, n)),
            None => out.push(t),
        }
    }
    Ok(out)
}

/// Images arrive as `data:<mime>;base64,<...>` URLs (no remote fetch).
fn image_bytes(part: &Value) -> Result<Vec<u8>, String> {
    let url = part
        .get("image_url")
        .and_then(|u| u.get("url").or(Some(u)))
        .and_then(|u| u.as_str())
        .ok_or("image_url part without a url")?;
    let b64 = url
        .strip_prefix("data:")
        .and_then(|r| r.split_once(";base64,"))
        .map(|(_, d)| d)
        .ok_or("image_url must be a data:<mime>;base64,... URL (remote URLs are not fetched)")?;
    base64_decode(b64)
}

/// Standard base64 (padding and whitespace tolerated).
pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=') {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return Err(format!("invalid base64 byte {c:#x}")),
        };
        acc = acc << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

/// A ChatML prompt, the template this model's GGUF declares. Control tokens go in by
/// id (they are single vocabulary entries, not text). An image part becomes
/// `<|vision_start|><|image_pad|><|vision_end|>` in place (one image per request);
/// its bytes are returned for [`Backend::set_image`].
fn chat_prompt<B: Backend>(req: &Value, b: &B) -> Result<(Vec<u32>, Vec<Vec<u8>>), String> {
    let Some(Value::Array(msgs)) = req.get("messages") else {
        return Err("messages must be an array".into());
    };
    let im_start = b.special("<|im_start|>");
    let im_end = b.special("<|im_end|>");
    let mut out = Vec::new();
    let mut images = Vec::new();
    for m in msgs {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        if let Some(s) = im_start {
            out.push(s);
        }
        // OpenAI content is a string or an array of parts, in order
        let mut text = format!("{role}\n");
        match m.get("content") {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) => text += s,
            Some(Value::Array(parts)) => {
                for p in parts {
                    match p.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            text += p.get("text").and_then(|t| t.as_str()).unwrap_or("")
                        }
                        Some("image_url") => {
                            let ids = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
                                .map(|t| b.special(t).ok_or("the model has no vision tokens"));
                            images.push(image_bytes(p)?);
                            out.extend(b.tokenize(&std::mem::take(&mut text))?);
                            for id in ids {
                                out.push(id?);
                            }
                        }
                        t => {
                            return Err(format!(
                                "content part type {:?} is not supported",
                                t.unwrap_or("?")
                            ))
                        }
                    }
                }
            }
            Some(_) => return Err("content must be a string or an array of parts".into()),
        };
        out.extend(b.tokenize(&text)?);
        if let Some(e) = im_end {
            out.push(e);
        }
        out.extend(b.tokenize("\n")?);
    }
    if let Some(s) = im_start {
        out.push(s);
    }
    out.extend(b.tokenize("assistant\n")?);
    Ok((out, images))
}

/// The text of the last `user` message (string content or its text parts).
fn last_user_text(req: &Value) -> Option<String> {
    let msgs = req.get("messages")?.as_array()?;
    let m = msgs.iter().rev().find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))?;
    match m.get("content")? {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

/// Whether a request asks for code: a fence, or a make-verb plus a code noun in the
/// same text. The MTP draft accepts ~88 % of code tokens but ~48 % of prose (11-08),
/// so speculative decoding only pays on code. Reading code to explain it is prose.
pub fn wants_code(text: &str) -> bool {
    let text = instruction(text);
    const VERBS: &[&str] = &[
        "write", "implement", "refactor", "fix", "create", "generate", "complete", "add", "rewrite",
        "port", "convert",
    ];
    const NOUNS: &[&str] = &[
        "function", "code", "script", "class", "method", "program", "snippet", "regex", "query", "patch",
        "diff", "module", "struct", "endpoint", "sql", "json", "yaml", "python", "rust", "javascript",
        "typescript", "bash", "shell", "cli", "test", "tests", "unittest", "api",
    ];
    if text.contains("```") {
        return true;
    }
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|w| !w.is_empty()).collect();
    words.iter().any(|w| VERBS.contains(w)) && words.iter().any(|w| NOUNS.contains(w))
}

/// The part of a user message that says what to do: all of a short message, else its
/// last paragraph when that reads as an ask, else its first (cut to 400 chars), so a pasted document or source
/// file does not make a "summarise this" request look like a code request (11-08 smoke).
fn instruction(text: &str) -> String {
    const SHORT: usize = 1200;
    if text.len() <= SHORT {
        return text.to_string();
    }
    let paras: Vec<&str> = text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()).collect();
    let cut = |p: &str| p.chars().take(400).collect::<String>();
    match paras.as_slice() {
        [] => String::new(),
        [one] => cut(one),
        [first, .., last] => {
            // a fenced block or code-like line at the end is material, not the instruction
            let last_is_ask = last.len() <= 400 && !last.contains("```") && !last.contains(';') && !last.contains('{');
            if last_is_ask { last.to_string() } else { cut(first) }
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_id() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod wants_code_tests {
    use super::wants_code;

    #[test]
    fn code_requests_and_prose_requests() {
        assert!(wants_code("Write a Python function `merge_intervals(intervals)` that merges overlapping intervals."));
        assert!(wants_code("Fix this:\n```rust\nfn main() {}\n```"));
        assert!(wants_code("implement the endpoint in Rust"));
        assert!(!wants_code("Explain in plain English why the sky looks blue during the day."));
        assert!(!wants_code("Here is a Python test file. Explain step by step what these tests check."));
        assert!(!wants_code("Summarize the key findings in a short bullet list for a busy engineer."));
        assert!(!wants_code("Explain this code to me."));
        // 11-08 smoke: a pasted source document must not turn "summarise" into code
        let doc = "fn add(a: u32) -> u32 { a + 1 }\n```rust\nfn x() {}\n```\nWrite the function; create a class.\n\n".repeat(60);
        assert!(!wants_code(&format!("Read the text below, then summarise its main argument.\n\n{doc}")));
        assert!(!wants_code(&format!("{doc}Summarise the text above in three sentences.")));
        assert!(wants_code(&format!("{doc}Now write a Rust function that does the same with a HashMap.")));
        assert!(wants_code(&format!("Implement a Python script that parses this log.\n\n{doc}")));
    }
}
