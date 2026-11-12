//! Stdio MCP server (newline-delimited JSON-RPC 2.0): initialize, ping, tools/list,
//! tools/call. Lets any MCP client (Hermes, Claude Code, OpenClaw) use these tools.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use crate::{call, canonical, Ctx, Tool};

/// Serve until `input` closes.
pub fn serve(ctx: &Ctx, tools: &[Tool], input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(m) => handle(ctx, tools, &m),
            Err(e) => Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": e.to_string()}})),
        };
        if let Some(r) = reply {
            writeln!(output, "{r}")?;
            output.flush()?;
        }
    }
    Ok(())
}

fn handle(ctx: &Ctx, tools: &[Tool], m: &Value) -> Option<Value> {
    let id = m.get("id")?.clone(); // no id = notification, never answered
    let p = &m["params"];
    let result = match m["method"].as_str().unwrap_or("") {
        "initialize" => Ok(json!({
            "protocolVersion": p["protocolVersion"].as_str().unwrap_or("2025-06-18"),
            "capabilities": {"tools": {"listChanged": false}},
            "serverInfo": {"name": "breccia-tools", "version": env!("CARGO_PKG_VERSION")}})),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tools.iter().map(|t| json!({
            "name": t.name, "description": t.description, "inputSchema": (t.schema)()})).collect::<Vec<_>>()})),
        "tools/call" => {
            let name = p["name"].as_str().unwrap_or("");
            let out = call(ctx, tools, name, &p["arguments"]);
            let err = out.starts_with("error: ");
            let mut content = vec![json!({"type": "text", "text": out})];
            if !err && canonical(name) == "view_image" {
                if let Some(img) = crate::image_part(ctx, &p["arguments"]) {
                    content.push(json!({"type": "image", "data": img.1, "mimeType": img.0}));
                }
            }
            Ok(json!({"content": content, "isError": err}))
        }
        other => Err(json!({"code": -32601, "message": format!("method not found: {other}")})),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{enabled, Config};

    #[test]
    fn json_rpc_smoke() {
        let d = std::env::temp_dir().join(format!("breccia-tools-mcp-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("a.txt"), "hello\n").unwrap();
        let ctx = Ctx::new(Config::new(d.clone()));
        let tools = enabled(&ctx, None);
        let input = [
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}"#,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"read_file","arguments":{"path":"a.txt"}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"read_file","arguments":{"path":"../x"}}}"#,
            r#"{"jsonrpc":"2.0","id":5,"method":"nope"}"#,
            "not json",
        ]
        .join("\n");
        let mut out = Vec::new();
        serve(&ctx, &tools, input.as_bytes(), &mut out).unwrap();
        let rs: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(rs.len(), 6, "notification gets no reply");
        assert_eq!(rs[0]["result"]["protocolVersion"], "2025-03-26");
        assert!(rs[1]["result"]["tools"].as_array().unwrap().iter().any(|t| t["name"] == "grep" && t["inputSchema"]["type"] == "object"));
        assert_eq!(rs[2]["result"]["content"][0]["text"], "     1\thello\n");
        assert_eq!(rs[2]["result"]["isError"], false);
        assert_eq!(rs[3]["result"]["isError"], true);
        assert_eq!(rs[4]["error"]["code"], -32601);
        assert_eq!(rs[5]["error"]["code"], -32700);
        std::fs::remove_dir_all(d).ok();
    }
}
