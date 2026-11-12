//! Agent state tools: todo_write, memory (file-backed, Hermes MEMORY.md/USER.md style),
//! session_search, skills_list / skill_view (SKILL.md dirs), ask_user and delegate
//! (front-end hooks).

use std::path::{Path, PathBuf};

use regex::RegexBuilder;
use serde_json::{json, Value};

use crate::{opt_b, opt_s, opt_u, s, Args, Ctx, Gate, Tool};

pub fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "todo_write",
            description: "Keep a task list for multi-step work. Send the full list (or merge=true to update items by id); no todos argument shows it.",
            schema: || json!({"type": "object", "properties": {
                "todos": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "content": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]}}, "required": ["content", "status"]}},
                "merge": {"type": "boolean"}}}),
            gate: Gate::Read,
            keys: &[("items", "todos"), ("tasks", "todos"), ("todo_list", "todos")],
            run: todo_write,
        },
        Tool {
            name: "memory",
            description: "Durable notes across sessions. action add|replace|remove|read; target memory (facts about work, environment) or user (preferences). replace/remove find the entry containing old.",
            schema: || json!({"type": "object", "properties": {
                "action": {"type": "string", "enum": ["add", "replace", "remove", "read"]},
                "target": {"type": "string", "enum": ["memory", "user"]},
                "content": {"type": "string"}, "old": {"type": "string"}}, "required": ["action"]}),
            gate: Gate::Write,
            keys: &[("old_text", "old"), ("match", "old"), ("text", "content"), ("new_text", "content"), ("operation", "action"), ("file", "target")],
            run: memory,
        },
        Tool {
            name: "session_search",
            description: "Search past agent sessions (regex, case-insensitive); returns matching messages with their session.",
            schema: || json!({"type": "object", "properties": {
                "query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}),
            gate: Gate::Read,
            keys: &[("pattern", "query"), ("q", "query"), ("max_results", "limit")],
            run: session_search,
        },
        Tool {
            name: "skills_list",
            description: "List available skills (name and description). Load one with skill_view before doing the task it covers.",
            schema: || json!({"type": "object", "properties": {"filter": {"type": "string"}}}),
            gate: Gate::Read,
            keys: &[("query", "filter"), ("category", "filter")],
            run: skills_list,
        },
        Tool {
            name: "skill_view",
            description: "Read a skill's SKILL.md, or another file inside that skill (file=relative path).",
            schema: || json!({"type": "object", "properties": {
                "name": {"type": "string"}, "file": {"type": "string"}}, "required": ["name"]}),
            gate: Gate::Read,
            keys: &[("skill", "name"), ("skill_name", "name"), ("path", "file"), ("file_path", "file")],
            run: skill_view,
        },
        Tool {
            name: "ask_user",
            description: "Ask the user a question when you cannot proceed without their decision. Give options when there are clear choices.",
            schema: || json!({"type": "object", "properties": {
                "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}, "required": ["question"]}),
            gate: Gate::Hook,
            keys: &[("prompt", "question"), ("message", "question"), ("choices", "options")],
            run: |ctx, a| ctx.ask.as_ref().ok_or("no user is attached")?(a),
        },
        Tool {
            name: "delegate",
            description: "Hand a self-contained subtask to a sub-agent with the same tools; it returns a summary. Put all needed context in task.",
            schema: || json!({"type": "object", "properties": {
                "task": {"type": "string"}, "context": {"type": "string"}}, "required": ["task"]}),
            gate: Gate::Hook,
            keys: &[("goal", "task"), ("prompt", "task"), ("description", "task"), ("instructions", "task")],
            run: |ctx, a| ctx.delegate.as_ref().ok_or("delegation is not available")?(a),
        },
    ]
}

fn todo_write(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let mut cur = ctx.todos.lock().unwrap();
    if let Some(new) = a.get("todos") {
        let new = match new {
            Value::String(s) => serde_json::from_str(s).map_err(|e| format!("todos: {e}"))?,
            v => v.clone(),
        };
        let items = new.as_array().ok_or("todos must be an array")?;
        let norm = |i: usize, t: &Value| {
            let g = |k: &[&str]| k.iter().find_map(|k| t.get(*k).and_then(Value::as_str)).unwrap_or("").to_string();
            let id = t.get("id").map(|v| v.as_str().map_or(v.to_string(), str::to_string)).unwrap_or_else(|| (i + 1).to_string());
            json!({"id": id, "content": g(&["content", "task", "title", "text", "description"]), "status": Some(g(&["status", "state"])).filter(|s| !s.is_empty()).unwrap_or("pending".into())})
        };
        if opt_b(a, "merge") {
            let list = cur.as_array_mut().unwrap();
            for (i, t) in items.iter().enumerate() {
                let t = norm(list.len() + i, t);
                match list.iter_mut().find(|x| x["id"] == t["id"]) {
                    Some(x) => {
                        if !t["content"].as_str().unwrap_or("").is_empty() {
                            x["content"] = t["content"].clone();
                        }
                        x["status"] = t["status"].clone();
                    }
                    None => list.push(t),
                }
            }
        } else {
            *cur = Value::Array(items.iter().enumerate().map(|(i, t)| norm(i, t)).collect());
        }
    }
    let list = cur.as_array().unwrap();
    if list.is_empty() {
        return Ok("(no todos)".into());
    }
    Ok(list
        .iter()
        .map(|t| {
            let m = match t["status"].as_str() {
                Some("completed") | Some("done") => "[x]",
                Some("in_progress") => "[>]",
                Some("cancelled") => "[-]",
                _ => "[ ]",
            };
            format!("{m} {}. {}", t["id"].as_str().unwrap_or(""), t["content"].as_str().unwrap_or(""))
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

const SEP: &str = "\n§\n";

fn memory_file(ctx: &Ctx, target: &str) -> Result<(PathBuf, usize), String> {
    let (f, cap) = match target {
        "memory" => ("MEMORY.md", 4000),
        "user" => ("USER.md", 2000),
        t => return Err(format!("target must be memory or user, not {t:?}")),
    };
    Ok((ctx.cfg.data_dir.join("memory").join(f), cap))
}

/// Entries of the memory/user file (for the agent's system prompt too).
pub fn memory_entries(ctx: &Ctx, target: &str) -> Vec<String> {
    let Ok((p, _)) = memory_file(ctx, target) else { return Vec::new() };
    std::fs::read_to_string(p).unwrap_or_default().split(SEP).map(str::trim).filter(|e| !e.is_empty()).map(str::to_string).collect()
}

fn memory(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let target = opt_s(a, "target").unwrap_or("memory");
    let (path, cap) = memory_file(ctx, target)?;
    let mut es = memory_entries(ctx, target);
    let find = |es: &[String]| -> Result<usize, String> {
        let old = s(a, "old")?;
        let hits: Vec<usize> = (0..es.len()).filter(|&i| es[i].contains(old)).collect();
        match hits[..] {
            [i] => Ok(i),
            [] => Err(format!("no {target} entry contains {old:?}")),
            _ => Err(format!("{} entries contain {old:?}; be more specific", hits.len())),
        }
    };
    match s(a, "action")? {
        "read" => return Ok(if es.is_empty() { format!("({target} is empty)") } else { es.join(SEP) }),
        "add" => es.push(s(a, "content")?.trim().to_string()),
        "replace" => {
            let i = find(&es)?;
            es[i] = s(a, "content")?.trim().to_string();
        }
        "remove" => {
            let i = find(&es)?;
            es.remove(i);
        }
        x => return Err(format!("unknown action {x:?}")),
    }
    let text = es.join(SEP);
    if text.chars().count() > cap {
        return Err(format!("{target} would be {} chars, over the {cap} cap; replace or remove entries first", text.chars().count()));
    }
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&path, &text).map_err(|e| e.to_string())?;
    Ok(format!("{target}: {} entries, {}/{cap} chars", es.len(), text.chars().count()))
}

fn session_search(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let re = RegexBuilder::new(s(a, "query")?).case_insensitive(true).size_limit(1 << 22).build().map_err(|e| e.to_string())?;
    let limit = opt_u(a, "limit").unwrap_or(20) as usize;
    let dir = ctx.cfg.data_dir.join("sessions");
    let mut files: Vec<_> = std::fs::read_dir(&dir).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    files.sort();
    files.reverse(); // newest first (names start with a timestamp)
    let mut out = Vec::new();
    for f in files {
        let name = f.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        for line in std::fs::read_to_string(&f).unwrap_or_default().lines() {
            let Ok(m) = serde_json::from_str::<Value>(line) else { continue };
            let text = match &m["content"] {
                Value::String(s) => s.clone(),
                Value::Null => continue,
                v => v.to_string(),
            };
            if let Some(hit) = re.find(&text) {
                let lo = text.floor_char_boundary(hit.start().saturating_sub(150));
                let hi = text.ceil_char_boundary((hit.end() + 150).min(text.len()));
                out.push(format!("[{name}] {}: …{}…", m["role"].as_str().unwrap_or("?"), text[lo..hi].replace('\n', " ")));
                if out.len() >= limit {
                    return Ok(out.join("\n"));
                }
            }
        }
    }
    Ok(if out.is_empty() { "no matches".into() } else { out.join("\n") })
}

/// (name, description, dir) for every `SKILL.md` up to three levels under the skill dirs.
fn skills(ctx: &Ctx) -> Vec<(String, String, PathBuf)> {
    fn scan(d: &Path, depth: u32, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
                if p.join("SKILL.md").is_file() {
                    out.push(p);
                } else if depth > 0 {
                    scan(&p, depth - 1, out);
                }
            }
        }
    }
    let mut dirs = Vec::new();
    for d in &ctx.cfg.skill_dirs {
        scan(d, 2, &mut dirs);
    }
    let mut v: Vec<_> = dirs
        .into_iter()
        .map(|d| {
            let (name, desc) = frontmatter(&std::fs::read_to_string(d.join("SKILL.md")).unwrap_or_default());
            let name = name.unwrap_or_else(|| d.file_name().unwrap().to_string_lossy().into_owned());
            (name, desc.unwrap_or_default(), d)
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v.dedup_by(|a, b| a.0 == b.0);
    v
}

/// `name` and `description` from YAML frontmatter (plain, quoted or `>`/`|` blocks).
fn frontmatter(t: &str) -> (Option<String>, Option<String>) {
    let mut ls = t.lines();
    if ls.next().map(str::trim) != Some("---") {
        return (None, None);
    }
    let body: Vec<&str> = ls.take_while(|l| l.trim() != "---").collect();
    let get = |k: &str| {
        let i = body.iter().position(|l| l.strip_prefix(k).is_some_and(|r| r.starts_with(':')))?;
        let v = body[i][k.len() + 1..].trim();
        let v = if v.is_empty() || v.starts_with('>') || v.starts_with('|') {
            body[i + 1..].iter().take_while(|l| l.starts_with(' ') || l.is_empty()).map(|l| l.trim()).collect::<Vec<_>>().join(" ")
        } else {
            v.trim_matches(|c| c == '"' || c == '\'').to_string()
        };
        Some(v.trim().to_string())
    };
    (get("name"), get("description"))
}

fn skills_list(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let f = opt_s(a, "filter").map(str::to_lowercase);
    let rows: Vec<String> = skills(ctx)
        .into_iter()
        .filter(|(n, d, _)| f.as_ref().is_none_or(|f| n.to_lowercase().contains(f) || d.to_lowercase().contains(f)))
        .map(|(n, d, _)| format!("{n} — {}", d.chars().take(200).collect::<String>()))
        .collect();
    Ok(if rows.is_empty() { "no skills found".into() } else { rows.join("\n") })
}

fn skill_view(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let want = s(a, "name")?;
    let all = skills(ctx);
    let (_, _, dir) = all
        .iter()
        .find(|(n, _, d)| n == want || d.file_name().is_some_and(|f| f == want))
        .ok_or_else(|| format!("no skill {want:?}; see skills_list"))?;
    let file = opt_s(a, "file").unwrap_or("SKILL.md");
    let base = dir.canonicalize().map_err(|e| e.to_string())?;
    let p = base.join(file).canonicalize().map_err(|e| format!("{file}: {e}"))?;
    if !p.starts_with(&base) {
        return Err(format!("{file} is outside the skill directory"));
    }
    std::fs::read_to_string(&p).map_err(|e| format!("{file}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{call, enabled, Config, Ctx};

    #[test]
    fn todo_memory_skills() {
        let d = std::env::temp_dir().join(format!("breccia-tools-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("skills/cat/demo")).unwrap();
        std::fs::write(d.join("skills/cat/demo/SKILL.md"), "---\nname: demo-skill\ndescription: >\n  Does a\n  demo.\n---\nbody\n").unwrap();
        std::fs::write(d.join("skills/cat/demo/ref.md"), "ref\n").unwrap();
        let mut c = Config::new(d.clone());
        c.data_dir = d.join("data");
        c.skill_dirs = vec![d.join("skills")];
        c.allow_write = true;
        let c = Ctx::new(c);
        let t = enabled(&c, None);
        let r = call(&c, &t, "todowrite", &json!({"todos": [{"content": "a", "status": "in_progress"}, {"content": "b", "status": "pending"}]}));
        assert_eq!(r, "[>] 1. a\n[ ] 2. b");
        let r = call(&c, &t, "todo_write", &json!({"todos": [{"id": "1", "status": "completed"}], "merge": true}));
        assert_eq!(r, "[x] 1. a\n[ ] 2. b");
        assert!(call(&c, &t, "memory", &json!({"action": "add", "content": "repo uses cargo"})).contains("1 entries"));
        assert!(call(&c, &t, "memory", &json!({"action": "replace", "old_text": "cargo", "content": "repo uses cargo + flock"})).contains("1 entries"));
        assert_eq!(memory_entries(&c, "memory"), vec!["repo uses cargo + flock"]);
        assert!(call(&c, &t, "memory", &json!({"action": "add", "content": "x".repeat(5000)})).contains("over the 4000 cap"));
        assert_eq!(call(&c, &t, "skills", &json!({})), "demo-skill — Does a demo.");
        assert_eq!(call(&c, &t, "skill_view", &json!({"name": "demo"})), "---\nname: demo-skill\ndescription: >\n  Does a\n  demo.\n---\nbody\n");
        assert_eq!(call(&c, &t, "skill", &json!({"name": "demo-skill", "file": "ref.md"})), "ref\n");
        assert!(call(&c, &t, "skill_view", &json!({"name": "demo", "file": "../../../skills/cat/demo/../../../data"})).starts_with("error"));
        std::fs::create_dir_all(d.join("data/sessions")).unwrap();
        std::fs::write(d.join("data/sessions/20261112-1.jsonl"), "{\"role\":\"user\",\"content\":\"fix the Flock bug\"}\n").unwrap();
        assert_eq!(call(&c, &t, "session_search", &json!({"query": "flock"})), "[20261112-1] user: …fix the Flock bug…");
        assert!(call(&c, &t, "ask_user", &json!({"question": "?"})).starts_with("error: unknown or disabled tool"));
        std::fs::remove_dir_all(d).ok();
    }
}
