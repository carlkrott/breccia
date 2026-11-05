//! `breccia-server` — an OpenAI-compatible HTTP surface over one generation session.
//!
//! The engine is one session that decodes one request at a time, so the server is a
//! single-threaded HTTP/1.1 loop: accept, read one request, generate, respond. That
//! is the honest shape for this backend (a thread pool would only queue on the same
//! session), and it keeps the crate dependency-free apart from `serde_json`.
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

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

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
        "breccia-server: listening on http://{addr} (model {})",
        backend.model_name()
    );
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
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .ok();
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);

    let mut line = String::new();
    if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
        return Ok(());
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

    let resp = dispatch(&method, &path, &body, backend);
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
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Route one request (exposed so the HTTP surface is testable without a model).
pub fn dispatch<B: Backend>(method: &str, path: &str, body: &[u8], b: &mut B) -> Response {
    let path = path.split('?').next().unwrap_or(path);
    match (method, path) {
        ("GET", "/health") => Response::json(
            200,
            json!({"status": "ok", "model": b.model_name(), "backend": "breccia"}),
        ),
        ("GET", "/v1/models") => {
            let id = b.model_name();
            Response::json(
                200,
                json!({"object": "list", "data": [{"id": id, "object": "model", "owned_by": "local"}]}),
            )
        }
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
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;

    let prompt_ids = if chat {
        match chat_prompt(&req, b).and_then(|(ids, imgs)| with_images(ids, imgs, b)) {
            Ok(ids) => ids,
            Err(e) => return Response::json(400, json!({"error": {"message": e}})),
        }
    } else {
        b.clear_images(); // images only ever come with the chat request that sent them
        match req.get("prompt") {
            Some(Value::String(s)) => match b.tokenize(s) {
                Ok(ids) => ids,
                Err(e) => return Response::json(500, json!({"error": {"message": e}})),
            },
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect(),
            _ => {
                return Response::json(
                    400,
                    json!({"error": {"message": "prompt must be a string or an id array"}}),
                )
            }
        }
    };
    if prompt_ids.is_empty() {
        return Response::json(400, json!({"error": {"message": "empty prompt"}}));
    }

    let t0 = std::time::Instant::now();
    let ids = match b.complete(&prompt_ids, max_tokens) {
        Ok(ids) => ids,
        Err(e) => return Response::json(500, json!({"error": {"message": e}})),
    };
    let seconds = t0.elapsed().as_secs_f64();
    let (ids, finish) = match b.eos().and_then(|eos| ids.iter().position(|&t| t == eos)) {
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
                    "finish_reason": finish,
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
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
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
