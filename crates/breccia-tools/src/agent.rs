//! OpenAI-compatible agent loop: send the conversation plus the fixed `tools` array, run
//! the returned tool_calls, repeat until the model answers without tools.
//!
//! Prefix-cache rules (Breccia reuses the KV of an unchanged prompt prefix): the system
//! prompt and tools array are fixed for the session, and assistant turns are echoed back
//! exactly as returned (`content`, `reasoning_content`, `tool_calls`).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::{call, canonical, enabled, image_part, openai_tools, Config, Ctx};

#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// e.g. `http://127.0.0.1:8165/v1` (a Breccia serve with BRECCIA_JINJA=1).
    pub base_url: String,
    pub model: String,
    /// File holding the API key (never on the command line).
    pub api_key_file: Option<PathBuf>,
    pub max_steps: usize,
    pub max_tokens: u64,
    /// Sent only when set; the model's template must accept the value.
    pub reasoning_effort: Option<String>,
    pub temperature: Option<f64>,
    /// Canonical tool names to offer (default: all the gates allow).
    pub only: Option<Vec<String>>,
    /// Extra system prompt text.
    pub system: Option<String>,
    /// HTTP timeout per call, seconds (long prefills).
    pub timeout: u64,
    pub verbose: bool,
}

impl AgentConfig {
    pub fn new(base_url: &str) -> AgentConfig {
        AgentConfig {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: "breccia".into(),
            api_key_file: None,
            max_steps: 50,
            max_tokens: 8192,
            reasoning_effort: None,
            temperature: None,
            only: None,
            system: None,
            timeout: 1800,
            verbose: false,
        }
    }
}

fn post(cfg: &AgentConfig, body: &Value) -> Result<Value, String> {
    let url = format!("{}/chat/completions", cfg.base_url);
    let mut cmd = Command::new("curl");
    cmd.args(["-sS", "--max-time", &cfg.timeout.to_string(), "-H", "Content-Type: application/json", "--data-binary", "@-", "-w", "\n%{http_code}"]);
    let hdr = crate::tmp_path("auth");
    if let Some(f) = &cfg.api_key_file {
        let key = std::fs::read_to_string(f).map_err(|e| format!("api key file {}: {e}", f.display()))?;
        let mut o = std::fs::OpenOptions::new();
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        o.write(true).create_new(true).open(&hdr).and_then(|mut h| writeln!(h, "Authorization: Bearer {}", key.trim())).map_err(|e| e.to_string())?;
        cmd.arg("-H").arg(format!("@{}", hdr.display()));
    }
    let mut child = cmd.arg(&url).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("curl: {e}"))?;
    let body = body.to_string();
    let w = child.stdin.take().map(|mut i| i.write_all(body.as_bytes()));
    let out = child.wait_with_output();
    let _ = std::fs::remove_file(&hdr);
    let out = out.map_err(|e| e.to_string())?;
    if let Some(Err(e)) = w {
        return Err(format!("sending request: {e}"));
    }
    if !out.status.success() {
        return Err(format!("{url}: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (resp, code) = text.rsplit_once('\n').unwrap_or((&text, "0"));
    let v: Value = serde_json::from_str(resp).map_err(|_| format!("HTTP {code} from {url}: {}", resp.chars().take(500).collect::<String>()))?;
    if code.trim() != "200" || v.get("error").is_some() {
        return Err(format!("HTTP {code} from {url}: {}", v.get("error").unwrap_or(&v)));
    }
    Ok(v)
}

fn system_prompt(ctx: &Ctx, cfg: &AgentConfig, names: &[&str]) -> String {
    let mut s = format!(
        "You are a capable autonomous agent working in {}. Use the tools to inspect and change things; do not guess file contents. \
         Prefer small, verifiable steps. When the task is done, answer with a short summary and no tool call.",
        ctx.cfg.root.display()
    );
    if !names.contains(&"write_file") {
        s.push_str(" You have read-only access: report what should change instead of changing it.");
    }
    for (t, head) in [("user", "About the user"), ("memory", "Your notes from earlier sessions")] {
        let es = crate::state::memory_entries(ctx, t);
        if names.contains(&"memory") && !es.is_empty() {
            s.push_str(&format!("\n\n## {head}\n- {}", es.join("\n- ")));
        }
    }
    if let Some(x) = &cfg.system {
        s.push_str("\n\n");
        s.push_str(x);
    }
    s
}

/// Run `task` to completion; returns the final answer.
pub fn run(tcfg: Config, cfg: &AgentConfig, task: &str) -> Result<String, String> {
    run_depth(tcfg, cfg, task, 0)
}

fn run_depth(tcfg: Config, cfg: &AgentConfig, task: &str, depth: u32) -> Result<String, String> {
    let mut ctx = Ctx::new(tcfg.clone());
    if depth == 0 {
        let (tc, ac) = (tcfg.clone(), Arc::new(cfg.clone()));
        ctx.delegate = Some(Box::new(move |a: &Map<String, Value>| {
            let task = a.get("task").and_then(Value::as_str).ok_or("missing task")?;
            let full = match a.get("context").and_then(Value::as_str) {
                Some(c) if !c.is_empty() => format!("{task}\n\nContext:\n{c}"),
                _ => task.to_string(),
            };
            run_depth(tc.clone(), &ac, &full, depth + 1).map(|r| format!("sub-agent result:\n{r}"))
        }));
        if unsafe { libc::isatty(0) } == 1 {
            ctx.ask = Some(Box::new(ask_stdin));
        }
    }
    let tools = enabled(&ctx, cfg.only.as_deref());
    let names: Vec<&str> = tools.iter().map(|t| t.name).collect();
    let tools_json = openai_tools(&tools);
    let mut messages = vec![json!({"role": "system", "content": system_prompt(&ctx, cfg, &names)}), json!({"role": "user", "content": task})];
    let sessions = tcfg.data_dir.join("sessions");
    let _ = std::fs::create_dir_all(&sessions);
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let log_path = sessions.join(format!("{secs}-{}-{depth}.jsonl", std::process::id()));
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path).ok();
    let mut record = |m: &Value| {
        if let Some(f) = log.as_mut() {
            let _ = writeln!(f, "{m}");
        }
    };
    messages.iter().for_each(&mut record);
    let say = |s: String| {
        if cfg.verbose {
            eprintln!("{}{s}", "  ".repeat(depth as usize));
        }
    };
    for step in 0..cfg.max_steps {
        let mut body = json!({"model": cfg.model, "messages": messages, "max_tokens": cfg.max_tokens, "stream": false});
        if !tools.is_empty() {
            body["tools"] = tools_json.clone();
            body["tool_choice"] = json!("auto");
        }
        if let Some(e) = &cfg.reasoning_effort {
            body["reasoning_effort"] = json!(e);
        }
        if let Some(t) = cfg.temperature {
            body["temperature"] = json!(t);
        }
        let t0 = std::time::Instant::now();
        let resp = post(cfg, &body)?;
        let msg = &resp["choices"][0]["message"];
        let finish = resp["choices"][0]["finish_reason"].as_str().unwrap_or("");
        // echo exactly what came back (prefix reuse)
        let mut echo = json!({"role": "assistant", "content": msg.get("content").cloned().unwrap_or(Value::Null)});
        for k in ["reasoning_content", "tool_calls"] {
            if let Some(v) = msg.get(k).filter(|v| !v.is_null()) {
                echo[k] = v.clone();
            }
        }
        record(&echo);
        messages.push(echo);
        let calls = msg["tool_calls"].as_array().cloned().unwrap_or_default();
        say(format!("[step {step}] {:.1}s finish={finish} tool_calls={} usage={}", t0.elapsed().as_secs_f64(), calls.len(), resp["usage"]));
        if calls.is_empty() {
            let text = msg["content"].as_str().unwrap_or("").to_string();
            if finish == "length" {
                return Err(format!("the answer hit max_tokens ({}); partial answer:\n{text}", cfg.max_tokens));
            }
            return Ok(text);
        }
        let mut images = Vec::new();
        for (i, c) in calls.iter().enumerate() {
            let f = &c["function"];
            let name = f["name"].as_str().unwrap_or("");
            let id = c["id"].as_str().map_or_else(|| format!("call_{step}_{i}"), str::to_string);
            say(format!("  -> {name} {}", f["arguments"].to_string().chars().take(200).collect::<String>()));
            let out = call(&ctx, &tools, name, &f["arguments"]);
            say(format!("  <- {}", out.lines().next().unwrap_or("").chars().take(160).collect::<String>()));
            if !out.starts_with("error: ") && canonical(name) == "view_image" {
                if let Some((mime, b64)) = image_part(&ctx, &f["arguments"]) {
                    images.push(json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{b64}")}}));
                }
            }
            let m = json!({"role": "tool", "tool_call_id": id, "name": name, "content": out});
            record(&m);
            messages.push(m);
        }
        if !images.is_empty() {
            // tool messages carry text only in most templates, so images ride in a user turn
            let mut parts = vec![json!({"type": "text", "text": "Images from view_image:"})];
            parts.extend(images);
            let m = json!({"role": "user", "content": parts});
            record(&json!({"role": "user", "content": "[images from view_image]"}));
            messages.push(m);
        }
    }
    Err(format!("stopped after {} steps without a final answer (session log {})", cfg.max_steps, log_path.display()))
}

fn ask_stdin(a: &Map<String, Value>) -> Result<String, String> {
    let q = a.get("question").and_then(Value::as_str).unwrap_or("?");
    eprintln!("\n[agent asks] {q}");
    if let Some(os) = a.get("options").and_then(Value::as_array) {
        for (i, o) in os.iter().enumerate() {
            eprintln!("  {}. {}", i + 1, o.as_str().unwrap_or(""));
        }
    }
    eprint!("> ");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
    let line = line.trim();
    let pick = line.parse::<usize>().ok().and_then(|n| a.get("options")?.as_array()?.get(n.checked_sub(1)?)?.as_str().map(str::to_string));
    Ok(format!("user answered: {}", pick.as_deref().unwrap_or(line)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read};
    use std::net::TcpListener;

    /// A two-step loop against a scripted OpenAI endpoint: tool call, then answer; checks the
    /// echoed assistant turn and that the tools array is identical across steps.
    #[test]
    fn loop_runs_tools_and_echoes() {
        let d = std::env::temp_dir().join(format!("breccia-tools-agent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("note.txt"), "the answer is 42\n").unwrap();
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for (n, s) in l.incoming().take(2).enumerate() {
                let mut s = s.unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut len = 0;
                loop {
                    let mut h = String::new();
                    r.read_line(&mut h).unwrap();
                    if h.trim().is_empty() {
                        break;
                    }
                    if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap();
                    }
                }
                let mut b = vec![0; len];
                r.read_exact(&mut b).unwrap();
                bodies.push(serde_json::from_slice::<Value>(&b).unwrap());
                let msg = if n == 0 {
                    json!({"role": "assistant", "content": "", "reasoning_content": "look at the file",
                        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{\"file_path\":\"note.txt\"}"}}]})
                } else {
                    json!({"role": "assistant", "content": "42"})
                };
                let out = json!({"choices": [{"message": msg, "finish_reason": if n == 0 {"tool_calls"} else {"stop"}}]}).to_string();
                write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}", out.len()).unwrap();
            }
            bodies
        });
        let mut tc = Config::new(d.clone());
        tc.data_dir = d.join("data");
        let r = run(tc, &AgentConfig::new(&format!("http://127.0.0.1:{port}/v1")), "what is the answer?").unwrap();
        assert_eq!(r, "42");
        let b = server.join().unwrap();
        assert_eq!(b[0]["tools"], b[1]["tools"]);
        assert!(b[0].get("reasoning_effort").is_none());
        let m = b[1]["messages"].as_array().unwrap();
        assert_eq!(m[2]["reasoning_content"], "look at the file");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"file_path\":\"note.txt\"}");
        assert_eq!(m[3]["role"], "tool");
        assert_eq!(m[3]["content"], "     1\tthe answer is 42\n");
        assert_eq!(m[..2], b[0]["messages"].as_array().unwrap()[..]);
        std::fs::remove_dir_all(d).ok();
    }
}
