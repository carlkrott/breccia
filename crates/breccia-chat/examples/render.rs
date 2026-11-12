//! `render REQ.json OUT_DIR TEMPLATE.jinja…`: render an OpenAI request through each chat
//! template into OUT_DIR/<stem>.txt ("ERROR: …" on failure). Golden checks against jinja2.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let req: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[0]).unwrap()).unwrap();
    for path in &a[2..] {
        let src = std::fs::read_to_string(path).unwrap();
        let out = match breccia_chat::ChatTemplate::new(&src, "<BOS>", "<EOS>").and_then(|t| t.render(&req)) {
            Ok(r) => r.text,
            Err(e) => format!("ERROR: {e}"),
        };
        let stem = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        std::fs::write(format!("{}/{stem}.txt", a[1]), out).unwrap();
    }
}
