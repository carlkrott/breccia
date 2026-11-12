//! Stdio MCP client: lets `breccia agent` offer the tools of external MCP servers
//! (kanban, browser, ...) next to its own, as `mcp__<server>__<tool>`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

pub struct McpServer {
    pub name: String,
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
    next_id: u64,
    timeout: Duration,
    /// `tools/list` entries as the server sent them.
    pub tools: Vec<Value>,
}

impl McpServer {
    /// Spawn `command` (run by `sh -c`, so it is trusted operator input), initialize, list tools.
    pub fn spawn(name: &str, command: &str, timeout: Duration) -> Result<McpServer, String> {
        let mut child = Command::new("sh")
            .args(["-c", command])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("mcp {name}: {e}"))?;
        let stdin = child.stdin.take().unwrap();
        let out = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for l in BufReader::new(out).lines().map_while(Result::ok) {
                if tx.send(l).is_err() {
                    break;
                }
            }
        });
        let mut s = McpServer { name: name.into(), child, stdin, rx, next_id: 1, timeout, tools: Vec::new() };
        s.request("initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "breccia-agent", "version": env!("CARGO_PKG_VERSION")}}))?;
        s.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))?;
        let mut cursor: Option<String> = None;
        loop {
            let r = s.request("tools/list", cursor.as_ref().map_or(json!({}), |c| json!({"cursor": c})))?;
            s.tools.extend(r["tools"].as_array().cloned().unwrap_or_default());
            match r["nextCursor"].as_str() {
                Some(c) if !c.is_empty() => cursor = Some(c.into()),
                _ => break,
            }
        }
        Ok(s)
    }

    fn send(&mut self, m: &Value) -> Result<(), String> {
        writeln!(self.stdin, "{m}").and_then(|_| self.stdin.flush()).map_err(|e| format!("mcp {}: {e}", self.name))
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        let end = Instant::now() + self.timeout;
        loop {
            let line = match self.rx.recv_timeout(end.saturating_duration_since(Instant::now())) {
                Ok(l) => l,
                Err(RecvTimeoutError::Timeout) => return Err(format!("mcp {}: {method} timed out", self.name)),
                Err(RecvTimeoutError::Disconnected) => return Err(format!("mcp {}: server exited", self.name)),
            };
            let Ok(m) = serde_json::from_str::<Value>(&line) else { continue };
            if m.get("method").is_some() {
                if let Some(rid) = m.get("id") {
                    // server-to-client requests (sampling, roots, elicitation) are not supported
                    self.send(&json!({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "not supported"}}))?;
                }
                continue;
            }
            if m["id"] != json!(id) {
                continue;
            }
            return match m.get("error") {
                Some(e) => Err(format!("mcp {}: {method}: {}", self.name, e["message"].as_str().unwrap_or(&e.to_string()))),
                None => Ok(m["result"].clone()),
            };
        }
    }

    /// OpenAI `tools` entries for this server's tools.
    pub fn openai_tools(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                json!({"type": "function", "function": {"name": format!("mcp__{}__{}", self.name, t["name"].as_str().unwrap_or("")),
                    "description": t["description"].as_str().unwrap_or(""),
                    "parameters": t.get("inputSchema").cloned().unwrap_or(json!({"type": "object"}))}})
            })
            .collect()
    }

    /// The server tool `full` (`mcp__<server>__<tool>`) names, if it is one of ours.
    pub fn owns<'a>(&self, full: &'a str) -> Option<&'a str> {
        let t = full.strip_prefix("mcp__")?.strip_prefix(self.name.as_str())?.strip_prefix("__")?;
        self.tools.iter().any(|x| x["name"] == t).then_some(t)
    }

    /// Call `tool`; text content joined, `error: ` prefix when the server flags an error.
    pub fn call(&mut self, tool: &str, args: &Value) -> String {
        let args = match args {
            Value::String(s) if s.trim().is_empty() => json!({}),
            Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
            Value::Null => json!({}),
            v => v.clone(),
        };
        match self.request("tools/call", json!({"name": tool, "arguments": args})) {
            Err(e) => format!("error: {e}"),
            Ok(r) => {
                let text: Vec<String> = r["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|c| match c["type"].as_str() {
                        Some("text") => c["text"].as_str().unwrap_or("").to_string(),
                        Some(k) => format!("[{k} content omitted]"),
                        None => String::new(),
                    })
                    .collect();
                let text = crate::cap(text.join("\n"), 50_000);
                if r["isError"] == true { format!("error: {text}") } else { text }
            }
        }
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talks_to_a_stdio_server() {
        let srv = r#"
import sys, json
for l in sys.stdin:
    m = json.loads(l)
    if "id" not in m or "method" not in m: continue
    r = {}
    if m["method"] == "tools/list":
        if m["params"].get("cursor"): r = {"tools": [{"name": "two", "inputSchema": {"type": "object"}}]}
        else: r = {"tools": [{"name": "echo", "description": "d", "inputSchema": {"type": "object"}}], "nextCursor": "p2"}
        print(json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"}))
        print(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "roots/list"}))
    if m["method"] == "tools/call":
        a = m["params"]["arguments"]
        r = {"content": [{"type": "text", "text": "got " + a.get("x", "")}], "isError": a.get("x") == "bad"}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#;
        let d = std::env::temp_dir().join(format!("breccia-mcpc-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("s.py"), srv).unwrap();
        let mut s = McpServer::spawn("k", &format!("python3 -u {}", d.join("s.py").display()), Duration::from_secs(10)).unwrap();
        assert_eq!(s.tools.len(), 2, "paginated tools/list");
        assert_eq!(s.openai_tools()[0]["function"]["name"], "mcp__k__echo");
        assert_eq!(s.owns("mcp__k__echo"), Some("echo"));
        assert_eq!(s.owns("mcp__k__nope"), None);
        assert_eq!(s.owns("mcp__kx__echo"), None);
        assert_eq!(s.call("echo", &json!(r#"{"x":"hi"}"#)), "got hi");
        assert_eq!(s.call("echo", &json!({"x": "bad"})), "error: got bad");
        drop(s);
        let mut dead = McpServer::spawn("z", "exit 0", Duration::from_secs(5));
        assert!(dead.as_mut().is_err());
        std::fs::remove_dir_all(d).ok();
    }
}
