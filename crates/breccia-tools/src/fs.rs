//! File tools: read_file, write_file, edit, apply_patch (see [`crate::patch`]), grep,
//! glob, list_dir, view_image. Every path resolves inside `Config::root`.

use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use regex::RegexBuilder;
use serde_json::{json, Value};

use crate::{opt_b, opt_s, opt_u, s, Args, Ctx, Gate, Tool};

const PATH_KEYS: &[(&str, &str)] =
    &[("file_path", "path"), ("filename", "path"), ("file", "path"), ("target_file", "path"), ("absolute_path", "path"), ("filePath", "path")];

pub fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "read_file",
            description: "Read a text file with line numbers. Use offset/limit (1-based lines) for large files.",
            schema: || json!({"type": "object", "properties": {
                "path": {"type": "string", "description": "File path, relative to the workdir or absolute inside it"},
                "offset": {"type": "integer", "description": "First line to show (1-based)"},
                "limit": {"type": "integer", "description": "Max lines (default 2000)"}}, "required": ["path"]}),
            gate: Gate::Read,
            keys: &[("file_path", "path"), ("filename", "path"), ("file", "path"), ("target_file", "path"), ("absolute_path", "path"), ("filePath", "path"),
                    ("start_line", "offset"), ("line", "offset"), ("max_lines", "limit"), ("lines", "limit")],
            run: read_file,
        },
        Tool {
            name: "write_file",
            description: "Create or overwrite a file with the given content. Parent directories are created.",
            schema: || json!({"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
            gate: Gate::Write,
            keys: &[("file_path", "path"), ("filename", "path"), ("file", "path"), ("target_file", "path"), ("absolute_path", "path"), ("filePath", "path"),
                    ("contents", "content"), ("text", "content"), ("file_text", "content")],
            run: write_file,
        },
        Tool {
            name: "edit",
            description: "Replace text in a file. old must match one place (whitespace-tolerant if no exact match) unless replace_all. edits=[{old,new}] applies several in order.",
            schema: || json!({"type": "object", "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Text to replace (include enough context to be unique)"},
                "new": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean"},
                "edits": {"type": "array", "items": {"type": "object", "properties": {"old": {"type": "string"}, "new": {"type": "string"}}}}},
                "required": ["path"]}),
            gate: Gate::Write,
            keys: &[("file_path", "path"), ("filename", "path"), ("file", "path"), ("target_file", "path"), ("absolute_path", "path"), ("filePath", "path"),
                    ("old_string", "old"), ("old_str", "old"), ("old_text", "old"), ("search", "old"), ("oldText", "old"),
                    ("new_string", "new"), ("new_str", "new"), ("new_text", "new"), ("replace", "new"), ("replacement", "new"), ("newText", "new"),
                    ("replaceAll", "replace_all"), ("all", "replace_all")],
            run: edit,
        },
        Tool {
            name: "apply_patch",
            description: "Apply a patch in the '*** Begin Patch' format: '*** Add File: p' (+lines), '*** Delete File: p', '*** Update File: p' (optional '*** Move to: q') with '@@' hunks of ' ', '-', '+' lines; ends with '*** End Patch'.",
            schema: || json!({"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}),
            gate: Gate::Write,
            keys: &[("input", "patch"), ("diff", "patch"), ("content", "patch")],
            run: crate::patch::run,
        },
        Tool {
            name: "grep",
            description: "Regex search in files under a directory (skips .git, target, node_modules, binaries). output_mode: content (default), files, count.",
            schema: || json!({"type": "object", "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Directory or file (default: workdir)"},
                "glob": {"type": "string", "description": "Only files whose path matches, e.g. *.rs or src/**/*.py"},
                "ignore_case": {"type": "boolean"},
                "context": {"type": "integer", "description": "Lines of context around each match"},
                "output_mode": {"type": "string", "enum": ["content", "files", "count"]},
                "limit": {"type": "integer", "description": "Max output lines (default 200)"}}, "required": ["pattern"]}),
            gate: Gate::Read,
            keys: &[("query", "pattern"), ("regex", "pattern"), ("search", "pattern"), ("dir", "path"), ("directory", "path"), ("file_path", "path"),
                    ("include", "glob"), ("file_glob", "glob"), ("file_pattern", "glob"), ("-i", "ignore_case"), ("case_insensitive", "ignore_case"),
                    ("-C", "context"), ("context_lines", "context"), ("max_results", "limit"), ("head_limit", "limit")],
            run: grep,
        },
        Tool {
            name: "glob",
            description: "Find files by glob pattern (*, ?, ** for any depth), newest first.",
            schema: || json!({"type": "object", "properties": {
                "pattern": {"type": "string"}, "path": {"type": "string", "description": "Base directory (default: workdir)"},
                "limit": {"type": "integer"}}, "required": ["pattern"]}),
            gate: Gate::Read,
            keys: &[("glob", "pattern"), ("query", "pattern"), ("name", "pattern"), ("dir", "path"), ("directory", "path"), ("max_results", "limit")],
            run: glob,
        },
        Tool {
            name: "list_dir",
            description: "List a directory: entries with type and size.",
            schema: || json!({"type": "object", "properties": {"path": {"type": "string", "description": "Default: workdir"}}}),
            gate: Gate::Read,
            keys: &[("dir", "path"), ("directory", "path"), ("dir_path", "path"), ("target_directory", "path")],
            run: list_dir,
        },
        Tool {
            name: "view_image",
            description: "Look at an image file (png, jpg, gif, webp); it is attached to the conversation.",
            schema: || json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
            gate: Gate::Read,
            keys: PATH_KEYS,
            run: view_image,
        },
    ]
}

/// `p` resolved inside the workdir. `p` may not exist yet (for writes): its nearest
/// existing ancestor is canonicalised (symlinks resolved) and the rest may hold no `..`.
pub fn resolve(ctx: &Ctx, p: &str) -> Result<PathBuf, String> {
    let root = ctx.cfg.root.canonicalize().map_err(|e| format!("workdir {}: {e}", ctx.cfg.root.display()))?;
    let p = p.strip_prefix("file://").unwrap_or(p);
    let full = if Path::new(p).is_absolute() { PathBuf::from(p) } else { root.join(p) };
    let mut existing = full.as_path();
    let mut rest = Vec::new();
    while existing.symlink_metadata().is_err() {
        rest.push(existing.file_name().ok_or("bad path")?);
        existing = existing.parent().ok_or("bad path")?;
    }
    let mut out = existing.canonicalize().map_err(|e| e.to_string())?;
    for c in rest.iter().rev() {
        if Path::new(c).components().any(|c| !matches!(c, Component::Normal(_))) {
            return Err(format!("{p}: '..' is not allowed in a new path"));
        }
        out.push(c);
    }
    if !out.starts_with(&root) {
        return Err(format!("{p} is outside the workdir {}", root.display()));
    }
    Ok(out)
}

fn read_text(path: &Path) -> Result<String, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.iter().take(8192).any(|&c| c == 0) {
        return Err(format!("{} is a binary file", path.display()));
    }
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn read_file(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let path = resolve(ctx, s(a, "path")?)?;
    if path.is_dir() {
        return list_dir(ctx, a);
    }
    let text = read_text(&path)?;
    let start = opt_u(a, "offset").unwrap_or(1).max(1) as usize;
    let limit = opt_u(a, "limit").unwrap_or(2000) as usize;
    let mut out = String::new();
    let mut n = 0;
    for (i, line) in text.lines().enumerate().skip(start - 1).take(limit) {
        let line = if line.len() > 2000 { &line[..line.floor_char_boundary(2000)] } else { line };
        out.push_str(&format!("{:>6}\t{line}\n", i + 1));
        n += 1;
    }
    let total = text.lines().count();
    if n == 0 {
        return Ok(if total == 0 { "(empty file)".into() } else { format!("(file has {total} lines; offset {start} is past the end)") });
    }
    if start - 1 + n < total {
        out.push_str(&format!("… ({total} lines; continue with offset {})\n", start + n));
    }
    Ok(out)
}

fn write_file(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let path = resolve(ctx, s(a, "path")?)?;
    let content = s(a, "content")?;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, content).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("wrote {} ({} bytes, {} lines)", path.display(), content.len(), content.lines().count()))
}

/// Byte ranges of `old` in `text`: exact first, then line-wise ignoring trailing, then
/// all surrounding whitespace (models often get indentation slightly wrong).
pub(crate) fn find_all(text: &str, old: &str) -> Vec<(usize, usize)> {
    let exact: Vec<_> = text.match_indices(old).map(|(i, m)| (i, i + m.len())).collect();
    if !exact.is_empty() || old.trim().is_empty() {
        return exact;
    }
    let want: Vec<&str> = old.trim_matches('\n').lines().collect();
    // line start offsets
    let mut starts = vec![0];
    starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
    let lines: Vec<&str> = text.lines().collect();
    for norm in [str::trim_end as fn(&str) -> &str, str::trim] {
        let mut hits = Vec::new();
        for i in 0..lines.len().saturating_sub(want.len() - 1) {
            if want.iter().enumerate().all(|(k, w)| norm(lines[i + k]) == norm(w)) {
                let end = starts[i + want.len() - 1] + lines[i + want.len() - 1].len();
                hits.push((starts[i], end));
            }
        }
        if !hits.is_empty() {
            return hits;
        }
    }
    Vec::new()
}

fn edit(ctx: &Ctx, a: &Args) -> Result<String, String> {
    // Hermes `patch` with V4A text in it
    if a.contains_key("patch") && !a.contains_key("old") {
        return crate::patch::run(ctx, a);
    }
    let path = resolve(ctx, s(a, "path")?)?;
    let mut text = read_text(&path)?;
    let pairs: Vec<(String, String)> = match a.get("edits").and_then(Value::as_array) {
        Some(es) => es
            .iter()
            .map(|e| {
                let g = |ks: &[&str]| ks.iter().find_map(|k| e.get(*k).and_then(Value::as_str)).map(str::to_string);
                Ok((
                    g(&["old", "old_string", "old_str", "oldText", "old_text"]).ok_or("edits[]: missing old")?,
                    g(&["new", "new_string", "new_str", "newText", "new_text"]).unwrap_or_default(),
                ))
            })
            .collect::<Result<_, String>>()?,
        None => vec![(s(a, "old")?.to_string(), opt_s(a, "new").unwrap_or("").to_string())],
    };
    let all = opt_b(a, "replace_all");
    let mut total = 0;
    for (old, new) in &pairs {
        if old.is_empty() {
            return Err("old is empty (use write_file to create a file)".into());
        }
        let hits = find_all(&text, old);
        match hits.len() {
            0 => return Err(format!("old text not found in {}; read the file and copy it exactly", path.display())),
            n if n > 1 && !all => return Err(format!("old text matches {n} places in {}; add context or set replace_all", path.display())),
            _ => {}
        }
        for (st, en) in hits.iter().rev() {
            text.replace_range(st..en, new);
        }
        total += hits.len();
    }
    std::fs::write(&path, &text).map_err(|e| e.to_string())?;
    Ok(format!("edited {}: {total} replacement(s)", path.display()))
}

/// `*` (not `/`), `?`, `**` (any depth, also zero dirs) over `/`-separated paths.
pub fn glob_match(pat: &str, path: &str) -> bool {
    fn m(p: &[u8], s: &[u8]) -> bool {
        match p.first() {
            None => s.is_empty(),
            // `**/` = zero or more directories; a bare `**` = anything, slashes included
            Some(b'*') if p.get(1) == Some(&b'*') && p.get(2) == Some(&b'/') => {
                m(&p[3..], s) || (0..s.len()).any(|i| s[i] == b'/' && m(&p[3..], &s[i + 1..]))
            }
            Some(b'*') if p.get(1) == Some(&b'*') => (0..=s.len()).any(|i| m(&p[2..], &s[i..])),
            Some(b'*') => (0..=s.len()).take_while(|&i| i == 0 || s[i - 1] != b'/').any(|i| m(&p[1..], &s[i..])),
            Some(b'?') => !s.is_empty() && s[0] != b'/' && m(&p[1..], &s[1..]),
            Some(c) => s.first() == Some(c) && m(&p[1..], &s[1..]),
        }
    }
    // a pattern without '/' matches the file name at any depth
    let target = if pat.contains('/') { path } else { path.rsplit('/').next().unwrap_or(path) };
    m(pat.as_bytes(), target.as_bytes())
}

const SKIP: &[&str] = &[".git", "target", "node_modules", "__pycache__", ".venv", ".hg", ".svn"];

/// Files under `dir` (recursive, sorted), skipping [`SKIP`] dirs; stops after `max`.
fn walk(dir: &Path, max: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut es: Vec<_> = rd.flatten().collect();
        es.sort_by_key(|e| std::cmp::Reverse(e.file_name()));
        for e in es {
            let Ok(ft) = e.file_type() else { continue };
            let name = e.file_name();
            if ft.is_dir() {
                if !SKIP.iter().any(|s| name == *s) {
                    stack.push(e.path());
                }
            } else if ft.is_file() {
                out.push(e.path());
                if out.len() >= max {
                    return out;
                }
            }
        }
    }
    out
}

fn rel<'a>(root: &Path, p: &'a Path) -> std::borrow::Cow<'a, str> {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy()
}

fn grep(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let re = RegexBuilder::new(s(a, "pattern")?)
        .case_insensitive(opt_b(a, "ignore_case"))
        .size_limit(1 << 22)
        .build()
        .map_err(|e| e.to_string())?;
    let base = resolve(ctx, opt_s(a, "path").unwrap_or("."))?;
    let root = ctx.cfg.root.canonicalize().map_err(|e| e.to_string())?;
    let ctxn = opt_u(a, "context").or_else(|| opt_u(a, "-A")).or_else(|| opt_u(a, "-B")).unwrap_or(0) as usize;
    let mode = opt_s(a, "output_mode").unwrap_or("content");
    let limit = opt_u(a, "limit").unwrap_or(200) as usize;
    let files = if base.is_file() { vec![base.clone()] } else { walk(&base, 100_000) };
    let mut out = Vec::new();
    let mut nfiles = 0;
    for f in files {
        let r = rel(&root, &f);
        if opt_s(a, "glob").is_some_and(|g| !glob_match(g, &r)) {
            continue;
        }
        if f.metadata().map_or(true, |m| m.len() > 4 << 20) {
            continue;
        }
        let Ok(text) = read_text(&f) else { continue };
        let lines: Vec<&str> = text.lines().collect();
        let hits: Vec<usize> = (0..lines.len()).filter(|&i| re.is_match(lines[i])).collect();
        if hits.is_empty() {
            continue;
        }
        nfiles += 1;
        match mode {
            "files" | "files_with_matches" => out.push(r.into_owned()),
            "count" => out.push(format!("{r}:{}", hits.len())),
            _ => {
                let mut last = None;
                for &h in &hits {
                    let (lo, hi) = (h.saturating_sub(ctxn), (h + ctxn).min(lines.len() - 1));
                    if ctxn > 0 && last.is_some_and(|l| lo > l + 1) {
                        out.push("--".into());
                    }
                    for i in lo.max(last.map_or(0, |l| l + 1))..=hi {
                        let sep = if i == h || hits.binary_search(&i).is_ok() { ':' } else { '-' };
                        let l = lines[i];
                        let l = if l.len() > 500 { &l[..l.floor_char_boundary(500)] } else { l };
                        out.push(format!("{r}{sep}{}{sep}{l}", i + 1));
                    }
                    last = Some(hi);
                }
            }
        }
        if out.len() >= limit {
            break;
        }
    }
    if out.is_empty() {
        return Ok("no matches".into());
    }
    let more = out.len() > limit;
    out.truncate(limit);
    let mut s = out.join("\n");
    if more || out.len() == limit {
        s.push_str(&format!("\n… (stopped at {limit} lines, {nfiles} files so far; narrow the pattern or raise limit)"));
    }
    Ok(s)
}

fn glob(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let pat = s(a, "pattern")?;
    let base = resolve(ctx, opt_s(a, "path").unwrap_or("."))?;
    let limit = opt_u(a, "limit").unwrap_or(200) as usize;
    let mut hits: Vec<(SystemTime, String)> = walk(&base, 200_000)
        .into_iter()
        .filter_map(|f| {
            let r = rel(&base, &f).into_owned();
            glob_match(pat, &r).then(|| (f.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH), r))
        })
        .collect();
    hits.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
    let n = hits.len();
    let mut s: Vec<String> = hits.into_iter().take(limit).map(|h| h.1).collect();
    if n > limit {
        s.push(format!("… ({n} files, showing {limit})"));
    }
    Ok(if s.is_empty() { "no files".into() } else { s.join("\n") })
}

fn list_dir(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let d = resolve(ctx, opt_s(a, "path").unwrap_or("."))?;
    let mut es: Vec<String> = std::fs::read_dir(&d)
        .map_err(|e| format!("{}: {e}", d.display()))?
        .flatten()
        .map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            match e.metadata() {
                Ok(m) if m.is_dir() => format!("{n}/"),
                Ok(m) => format!("{n}  ({} bytes)", m.len()),
                Err(_) => n,
            }
        })
        .collect();
    es.sort();
    let n = es.len();
    es.truncate(500);
    if n > 500 {
        es.push(format!("… ({n} entries)"));
    }
    Ok(if es.is_empty() { "(empty)".into() } else { es.join("\n") })
}

pub(crate) fn image_mime(p: &Path) -> Option<&'static str> {
    match p.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn view_image(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let p = resolve(ctx, s(a, "path")?)?;
    image_mime(&p).ok_or("not a png/jpg/gif/webp file")?;
    let n = std::fs::metadata(&p).map_err(|e| e.to_string())?.len();
    if n > 20 << 20 {
        return Err("image over 20 MB".into());
    }
    Ok(format!("image {} ({n} bytes) attached below", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{call, enabled, Config, Ctx};

    fn ctx(tag: &str) -> (Ctx, PathBuf) {
        let d = std::env::temp_dir().join(format!("breccia-tools-fs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("src/deep")).unwrap();
        let mut c = Config::new(d.clone());
        c.allow_write = true;
        (Ctx::new(c), d)
    }

    #[test]
    fn glob_patterns() {
        assert!(glob_match("*.rs", "src/deep/a.rs"));
        assert!(glob_match("src/**/*.rs", "src/a.rs"));
        assert!(glob_match("src/**/*.rs", "src/x/y/a.rs"));
        assert!(!glob_match("src/*.rs", "src/x/a.rs"));
        assert!(glob_match("**", "a/b"));
        assert!(glob_match("a?c.txt", "abc.txt"));
        assert!(glob_match("src/**", "src/x/y.rs"));
        assert!(!glob_match("src/**/*.rs", "lib/a.rs"));
    }

    #[test]
    fn paths_stay_inside_the_workdir() {
        let (c, d) = ctx("paths");
        assert!(resolve(&c, "../x").is_err());
        assert!(resolve(&c, "/etc/passwd").is_err());
        assert!(resolve(&c, "new/../../x").is_err());
        std::os::unix::fs::symlink("/etc", d.join("link")).unwrap();
        assert!(resolve(&c, "link/passwd").is_err());
        std::os::unix::fs::symlink("/tmp/breccia-nonexistent-target", d.join("dangling")).unwrap();
        assert!(resolve(&c, "dangling").is_err(), "a dangling symlink must not be writable through");
        assert!(resolve(&c, "dangling/x").is_err());
        assert!(resolve(&c, "src/new.txt").unwrap().starts_with(d.canonicalize().unwrap()));
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn edit_grep_glob_roundtrip() {
        let (c, d) = ctx("edit");
        let t = enabled(&c, None);
        let r = call(&c, &t, "write", &json!({"file_path": "src/deep/a.rs", "contents": "fn a() {\n    let x = 1;\n    let y = 2;\n}\n"}));
        assert!(r.starts_with("wrote"), "{r}");
        // fuzzy: wrong indentation still matches one place
        let r = call(&c, &t, "str_replace", &json!({"path": "src/deep/a.rs", "old_str": "let x = 1;\nlet y = 2;", "new_str": "    let z = 3;"}));
        assert!(r.contains("1 replacement"), "{r}");
        assert_eq!(std::fs::read_to_string(d.join("src/deep/a.rs")).unwrap(), "fn a() {\n    let z = 3;\n}\n");
        let r = call(&c, &t, "edit", &json!({"path": "src/deep/a.rs", "old": "nope", "new": "x"}));
        assert!(r.starts_with("error: edit: old text not found"), "{r}");
        let r = call(&c, &t, "grep", &json!({"pattern": "LET Z", "-i": true, "include": "*.rs"}));
        assert_eq!(r, "src/deep/a.rs:2:    let z = 3;");
        assert_eq!(call(&c, &t, "glob", &json!({"pattern": "**/*.rs"})), "src/deep/a.rs");
        assert_eq!(call(&c, &t, "ls", &json!({"path": "src"})), "deep/");
        std::fs::remove_dir_all(d).ok();
    }
}
