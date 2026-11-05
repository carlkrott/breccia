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
        match chat_prompt(&req, b) {
            Ok(ids) => ids,
            Err(e) => return Response::json(400, json!({"error": {"message": e}})),
        }
    } else {
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

/// A ChatML prompt, the template this model's GGUF declares. Control tokens go in by
/// id (they are single vocabulary entries, not text).
fn chat_prompt<B: Backend>(req: &Value, b: &B) -> Result<Vec<u32>, String> {
    let Some(Value::Array(msgs)) = req.get("messages") else {
        return Err("messages must be an array".into());
    };
    let im_start = b.special("<|im_start|>");
    let im_end = b.special("<|im_end|>");
    let mut out = Vec::new();
    for m in msgs {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        // OpenAI content is a string or an array of parts; image parts must fail loudly,
        // not run as an empty prompt.
        let content = match m.get("content") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for p in parts {
                    match p.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            text += p.get("text").and_then(|t| t.as_str()).unwrap_or("")
                        }
                        t => {
                            return Err(format!(
                                "content part type {:?} is not supported (no image input yet)",
                                t.unwrap_or("?")
                            ))
                        }
                    }
                }
                text
            }
            Some(_) => return Err("content must be a string or an array of parts".into()),
        };
        if let Some(s) = im_start {
            out.push(s);
        }
        out.extend(b.tokenize(&format!("{role}\n{content}"))?);
        if let Some(e) = im_end {
            out.push(e);
        }
        out.extend(b.tokenize("\n")?);
    }
    if let Some(s) = im_start {
        out.push(s);
    }
    out.extend(b.tokenize("assistant\n")?);
    Ok(out)
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
