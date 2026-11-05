//! The HTTP surface, without a model: routing, JSON shapes, and the chat
//! template's control-token placement.

use breccia_server::{dispatch, Backend};

struct Mock;

impl Backend for Mock {
    fn model_name(&self) -> String {
        "mock".into()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        // One id per byte, offset so 0 is never a token.
        Ok(text.bytes().map(|b| b as u32 + 1).collect())
    }
    fn decode_text(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| char::from_u32(i.saturating_sub(1)).unwrap_or('?'))
            .collect()
    }
    fn complete(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>, String> {
        assert!(!prompt.is_empty());
        Ok(format!("ok:{max_tokens}")
            .bytes()
            .map(|b| b as u32 + 1)
            .collect())
    }
    fn eos(&self) -> Option<u32> {
        None
    }
    fn special(&self, text: &str) -> Option<u32> {
        (text == "<|im_start|>").then_some(u32::MAX - 1)
    }
}

fn body(v: serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&v).unwrap()
}

#[test]
fn health_models_and_404() {
    let mut m = Mock;
    let r = dispatch("GET", "/health", b"", &mut m);
    assert_eq!(r.status, 200);
    let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!(v["status"], "ok");

    let r = dispatch("GET", "/v1/models", b"", &mut m);
    let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!(v["data"][0]["id"], "mock");

    assert_eq!(dispatch("GET", "/nope", b"", &mut m).status, 404);
    assert_eq!(dispatch("GET", "/v1/completions", b"", &mut m).status, 405);
}

#[test]
fn completion_reports_text_and_usage() {
    let mut m = Mock;
    let r = dispatch(
        "POST",
        "/v1/completions",
        &body(serde_json::json!({"prompt": "abc", "max_tokens": 5})),
        &mut m,
    );
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!(v["choices"][0]["text"], "ok:5");
    assert_eq!(v["usage"]["prompt_tokens"], 3);
    assert_eq!(v["usage"]["completion_tokens"], 4);
    assert_eq!(v["choices"][0]["finish_reason"], "length");

    // An id array is accepted as the prompt.
    let r = dispatch(
        "POST",
        "/v1/completions",
        &body(serde_json::json!({"prompt": [1, 2, 3], "max_tokens": 2})),
        &mut m,
    );
    assert_eq!(r.status, 200);

    // Bad JSON / empty prompt are 400s, not panics.
    assert_eq!(
        dispatch("POST", "/v1/completions", b"{nope", &mut m).status,
        400
    );
    assert_eq!(
        dispatch(
            "POST",
            "/v1/completions",
            &body(serde_json::json!({"prompt": "", "max_tokens": 2})),
            &mut m
        )
        .status,
        400
    );
}

#[test]
fn chat_puts_the_control_tokens_in_by_id() {
    let mut m = Mock;
    let r = dispatch(
        "POST",
        "/v1/chat/completions",
        &body(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3
        })),
        &mut m,
    );
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    // im_start(1) + "user\nhi"(7) + im_end absent + "\n"(1) + im_start(1)
    // + "assistant\n"(10) = 20 prompt tokens.
    assert_eq!(v["usage"]["prompt_tokens"], 20);
}

#[test]
fn chat_rejects_image_parts_and_joins_text_parts() {
    let mut m = Mock;
    let img = dispatch(
        "POST",
        "/v1/chat/completions",
        &body(serde_json::json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}],
            "max_tokens": 3
        })),
        &mut m,
    );
    assert_eq!(img.status, 400, "{}", String::from_utf8_lossy(&img.body));
    let text = dispatch(
        "POST",
        "/v1/chat/completions",
        &body(serde_json::json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 3
        })),
        &mut m,
    );
    let v: serde_json::Value = serde_json::from_slice(&text.body).unwrap();
    assert_eq!(
        v["usage"]["prompt_tokens"], 20,
        "array text must match the string form"
    );
}
