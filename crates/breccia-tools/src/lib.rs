//! Agent tools for Breccia: one Rust implementation of the tool union of Hermes,
//! OpenClaw, the DeepSeek harness and the Grok CLIs (files, edit/patch, search, shell,
//! processes, web, todo/memory/skills/sessions, ask/delegate), each exposed under one
//! canonical name. Other harnesses' names and argument keys are accepted at dispatch
//! ([`ALIASES`], [`Tool::keys`]), never listed, so the `tools` array (and the prompt
//! prefix it renders into) stays small and stable.
//!
//! Front ends: [`agent`] (an OpenAI-compatible loop against a Breccia serve or any
//! other endpoint) and [`mcp`] (a stdio MCP server). Tools never run inside serve.
//!
//! Safety: file tools are confined to the workdir (canonical paths, so `..` and
//! symlinks cannot leave it); writes and exec are off unless enabled; `process kill`
//! only signals process groups this crate spawned; `web_fetch` refuses private,
//! loopback, link-local and tailnet (100.64/10) addresses on every redirect hop.

use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::{json, Map, Value};

pub mod agent;
pub mod exec;
pub mod fs;
pub mod mcp;
pub mod patch;
pub mod state;
pub mod web;

/// What the tools may touch.
#[derive(Clone, Debug)]
pub struct Config {
    /// Workdir: every file path resolves inside it.
    pub root: PathBuf,
    pub allow_write: bool,
    pub allow_exec: bool,
    /// SearXNG base URL for `web_search`.
    pub searx: String,
    /// Agent state: `memory/`, `sessions/`.
    pub data_dir: PathBuf,
    /// Directories searched for `*/SKILL.md`.
    pub skill_dirs: Vec<PathBuf>,
    /// Cap on one tool result, in bytes.
    pub max_output: usize,
}

impl Config {
    pub fn new(root: PathBuf) -> Config {
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()));
        Config {
            root,
            allow_write: false,
            allow_exec: false,
            searx: std::env::var("BRECCIA_SEARX").unwrap_or_else(|_| "http://127.0.0.1:8888".into()),
            data_dir: home.join(".local/share/breccia/agent"),
            skill_dirs: vec![home.join(".hermes/skills"), home.join(".claude/skills")],
            max_output: 50_000,
        }
    }
}

type Hook = Box<dyn Fn(&Map<String, Value>) -> Result<String, String> + Send + Sync>;

/// Per-session tool state.
pub struct Ctx {
    pub cfg: Config,
    pub(crate) procs: Mutex<exec::Procs>,
    pub(crate) todos: Mutex<Value>,
    /// `ask_user`: only an interactive front end sets it.
    pub ask: Option<Hook>,
    /// `delegate`: the agent loop sets it to a nested run.
    pub delegate: Option<Hook>,
}

impl Ctx {
    pub fn new(cfg: Config) -> Ctx {
        Ctx { cfg, procs: Mutex::new(exec::Procs::default()), todos: Mutex::new(json!([])), ask: None, delegate: None }
    }
}

pub type Args = Map<String, Value>;
type Run = fn(&Ctx, &Args) -> Result<String, String>;

/// Which gate a tool sits behind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gate {
    Read,
    Write,
    Exec,
    /// Needs a front-end hook (`ask_user`, `delegate`).
    Hook,
}

pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub schema: fn() -> Value,
    pub gate: Gate,
    /// Other harnesses' argument keys → ours.
    pub keys: &'static [(&'static str, &'static str)],
    pub run: Run,
}

/// Other harnesses' tool names → ours (Hermes, OpenClaw, DeepSeek harness, Grok CLIs,
/// Claude Code / Codex spellings).
pub const ALIASES: &[(&str, &str)] = &[
    ("read", "read_file"), ("view", "read_file"), ("cat", "read_file"),
    ("write", "write_file"), ("create_file", "write_file"),
    ("patch", "edit"), ("edit_file", "edit"), ("search_replace", "edit"), ("str_replace", "edit"),
    ("str_replace_editor", "edit"), ("multi_edit", "edit"), ("multiedit", "edit"), ("replace", "edit"),
    ("applypatch", "apply_patch"),
    ("search_files", "grep"), ("grep_files", "grep"), ("grep_search", "grep"), ("search", "grep"),
    ("find", "glob"), ("find_files", "glob"), ("file_search", "glob"),
    ("ls", "list_dir"), ("list_directory", "list_dir"), ("list_files", "list_dir"),
    ("terminal", "bash"), ("exec", "bash"), ("shell", "bash"), ("run_terminal_cmd", "bash"),
    ("run_command", "bash"), ("execute_command", "bash"), ("run_shell_command", "bash"),
    ("process_manage", "process"), ("process_logs", "process"), ("job", "process"),
    ("python", "execute_code"), ("code_execution", "execute_code"), ("run_code", "execute_code"),
    ("websearch", "web_search"), ("search_web", "web_search"),
    ("webfetch", "web_fetch"), ("web_extract", "web_fetch"), ("fetch", "web_fetch"), ("read_url", "web_fetch"),
    ("todowrite", "todo_write"), ("todo_list", "todo_write"), ("todo", "todo_write"), ("update_todos", "todo_write"),
    ("clarify", "ask_user"), ("ask_user_question", "ask_user"), ("ask_followup_question", "ask_user"),
    ("task", "delegate"), ("subagent", "delegate"), ("delegate_task", "delegate"), ("spawn_agent", "delegate"),
    ("skill", "skill_view"), ("skills_view", "skill_view"), ("skills", "skills_list"),
    ("search_sessions", "session_search"),
    ("image", "view_image"), ("vision_analyze", "view_image"), ("read_image", "view_image"),
];

/// Every tool, in the fixed order the `tools` array uses.
pub fn all() -> Vec<Tool> {
    let mut v = fs::tools();
    v.extend(exec::tools());
    v.extend(web::tools());
    v.extend(state::tools());
    v
}

/// The tools this context may run: gates from [`Config`], hooks present, and an
/// optional name filter (`only`, canonical names).
pub fn enabled(ctx: &Ctx, only: Option<&[String]>) -> Vec<Tool> {
    all()
        .into_iter()
        .filter(|t| match t.gate {
            Gate::Read => true,
            Gate::Write => ctx.cfg.allow_write,
            Gate::Exec => ctx.cfg.allow_exec,
            Gate::Hook => (t.name == "ask_user" && ctx.ask.is_some()) || (t.name == "delegate" && ctx.delegate.is_some()),
        })
        .filter(|t| only.is_none_or(|o| o.iter().any(|n| n == t.name)))
        .collect()
}

/// OpenAI `tools` array.
pub fn openai_tools(tools: &[Tool]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": (t.schema)()}}))
            .collect(),
    )
}

/// Canonical name for `name` (`mcp__server__tool` prefixes and case are ignored).
pub fn canonical(name: &str) -> String {
    let n = name.rsplit("__").next().unwrap_or(name).to_ascii_lowercase();
    ALIASES.iter().find(|(a, _)| *a == n).map_or(n, |(_, c)| c.to_string())
}

/// Run one tool call; `args` is the JSON arguments string (or object). Errors come
/// back as text for the model, never as a panic.
pub fn call(ctx: &Ctx, tools: &[Tool], name: &str, args: &Value) -> String {
    let canon = canonical(name);
    let Some(t) = tools.iter().find(|t| t.name == canon) else {
        let names: Vec<&str> = tools.iter().map(|t| t.name).collect();
        return format!("error: unknown or disabled tool {name:?}; available: {}", names.join(", "));
    };
    let parsed = match args {
        Value::String(s) if s.trim().is_empty() => Value::Object(Map::new()),
        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        Value::Null => Value::Object(Map::new()),
        v => v.clone(),
    };
    let Value::Object(mut a) = parsed else {
        return format!("error: {canon}: arguments must be a JSON object");
    };
    for (from, to) in t.keys {
        if !a.contains_key(*to) {
            if let Some(v) = a.remove(*from) {
                a.insert(to.to_string(), v);
            }
        }
    }
    match (t.run)(ctx, &a) {
        Ok(s) => cap(s, ctx.cfg.max_output),
        Err(e) => format!("error: {canon}: {e}"),
    }
}

/// Keep at most `max` bytes, cut on a char boundary, with a note.
pub fn cap(mut s: String, max: usize) -> String {
    if s.len() > max {
        let mut i = max;
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        let total = s.len();
        s.truncate(i);
        s.push_str(&format!("\n… [truncated: {total} bytes, showing the first {i}]"));
    }
    s
}

/// (mime, base64) of the image a `view_image` call named, for the front end to attach.
pub fn image_part(ctx: &Ctx, args: &Value) -> Option<(&'static str, String)> {
    let a: Value = match args {
        Value::String(s) => serde_json::from_str(s).ok()?,
        v => v.clone(),
    };
    let p = ["path", "file_path", "image", "image_path", "file", "url"].iter().find_map(|k| a.get(*k)?.as_str())?;
    let p = fs::resolve(ctx, p).ok()?;
    Some((fs::image_mime(&p)?, base64(&std::fs::read(p).ok()?)))
}

pub fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut o = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            o.push(if i <= c.len() { T[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    o
}

/// A fresh path in the temp dir (unique per process and call).
pub(crate) fn tmp_path(tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!("breccia-{tag}-{}-{}", std::process::id(), SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)))
}

pub(crate) fn s<'a>(a: &'a Args, k: &str) -> Result<&'a str, String> {
    a.get(k).and_then(Value::as_str).ok_or_else(|| format!("missing string argument {k:?}"))
}
pub(crate) fn opt_s<'a>(a: &'a Args, k: &str) -> Option<&'a str> {
    a.get(k).and_then(Value::as_str).filter(|v| !v.is_empty())
}
/// Integer argument; models often send numbers as strings.
pub(crate) fn opt_u(a: &Args, k: &str) -> Option<u64> {
    a.get(k).and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)).or_else(|| v.as_str()?.trim().parse().ok()))
}
pub(crate) fn opt_b(a: &Args, k: &str) -> bool {
    a.get(k).is_some_and(|v| v.as_bool().unwrap_or_else(|| v.as_str() == Some("true")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_keys_normalise() {
        let dir = std::env::temp_dir().join(format!("breccia-tools-lib-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "x\ny\n").unwrap();
        let ctx = Ctx::new(Config::new(dir.clone()));
        let tools = enabled(&ctx, None);
        assert!(tools.iter().all(|t| t.gate == Gate::Read), "writes/exec are off by default");
        let out = call(&ctx, &tools, "mcp__fs__Read", &json!(r#"{"file_path": "a.txt", "offset": "2"}"#));
        assert_eq!(out, "     2\ty\n");
        assert!(call(&ctx, &tools, "bash", &json!({})).starts_with("error: unknown or disabled tool"));
        let names: Vec<_> = all().iter().map(|t| t.name).collect();
        for (a, c) in ALIASES {
            assert!(names.contains(c), "alias {a} -> missing {c}");
            assert!(!names.contains(a), "alias {a} shadows a tool");
        }
        std::fs::remove_dir_all(dir).ok();
    }
}
