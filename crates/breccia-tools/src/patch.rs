//! `apply_patch`: the "*** Begin Patch" (V4A) format used by Codex, Hermes and the Grok
//! CLIs. Every hunk is applied in memory first; nothing is written unless all apply.

use std::path::PathBuf;

use crate::fs::resolve;
use crate::{s, Args, Ctx};

enum Op {
    Add(PathBuf, String),
    Delete(PathBuf),
    Update { path: PathBuf, to: Option<PathBuf>, text: String },
}

pub fn run(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let ops = plan(ctx, s(a, "patch")?)?;
    let mut done = Vec::new();
    for op in ops {
        match op {
            Op::Add(p, t) => {
                if let Some(d) = p.parent() {
                    std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                }
                std::fs::write(&p, t).map_err(|e| format!("{}: {e}", p.display()))?;
                done.push(format!("A {}", p.display()));
            }
            Op::Delete(p) => {
                std::fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                done.push(format!("D {}", p.display()));
            }
            Op::Update { path, to, text } => {
                let dst = to.as_ref().unwrap_or(&path);
                if let Some(d) = dst.parent() {
                    std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                }
                std::fs::write(dst, text).map_err(|e| format!("{}: {e}", dst.display()))?;
                if to.is_some() {
                    std::fs::remove_file(&path).map_err(|e| e.to_string())?;
                    done.push(format!("R {} -> {}", path.display(), dst.display()));
                } else {
                    done.push(format!("M {}", path.display()));
                }
            }
        }
    }
    Ok(done.join("\n"))
}

/// Parse and apply in memory; errors name the file and hunk.
fn plan(ctx: &Ctx, patch: &str) -> Result<Vec<Op>, String> {
    let lines: Vec<&str> = patch.lines().map(|l| l.strip_suffix('\r').unwrap_or(l)).collect();
    let start = lines.iter().position(|l| l.trim() == "*** Begin Patch").ok_or("missing '*** Begin Patch'")?;
    let mut i = start + 1;
    let mut ops = Vec::new();
    let header = |l: &str| l.starts_with("*** ");
    while i < lines.len() {
        let l = lines[i];
        if l.trim() == "*** End Patch" {
            return if ops.is_empty() { Err("patch has no file operations".into()) } else { Ok(ops) };
        }
        if let Some(p) = l.strip_prefix("*** Add File: ") {
            let path = resolve(ctx, p.trim())?;
            if path.exists() {
                return Err(format!("Add File {}: already exists (use Update File)", p.trim()));
            }
            i += 1;
            let mut t = String::new();
            while i < lines.len() && !header(lines[i]) {
                t.push_str(lines[i].strip_prefix('+').ok_or_else(|| format!("Add File {}: line without '+': {:?}", p.trim(), lines[i]))?);
                t.push('\n');
                i += 1;
            }
            ops.push(Op::Add(path, t));
        } else if let Some(p) = l.strip_prefix("*** Delete File: ") {
            let path = resolve(ctx, p.trim())?;
            if !path.is_file() {
                return Err(format!("Delete File {}: no such file", p.trim()));
            }
            ops.push(Op::Delete(path));
            i += 1;
        } else if let Some(p) = l.strip_prefix("*** Update File: ") {
            let path = resolve(ctx, p.trim())?;
            let orig = std::fs::read_to_string(&path).map_err(|e| format!("Update File {}: {e}", p.trim()))?;
            i += 1;
            let mut to = None;
            if let Some(m) = lines.get(i).and_then(|l| l.strip_prefix("*** Move to: ")) {
                to = Some(resolve(ctx, m.trim())?);
                i += 1;
            }
            let mut body = Vec::new();
            while i < lines.len() && (!header(lines[i]) || lines[i].trim() == "*** End of File") {
                body.push(lines[i]);
                i += 1;
            }
            let text = apply(&orig, &body).map_err(|e| format!("Update File {}: {e}", p.trim()))?;
            ops.push(Op::Update { path, to, text });
        } else if l.trim().is_empty() {
            i += 1;
        } else {
            return Err(format!("unexpected line {:?} (expected a '*** Add/Delete/Update File:' header)", l));
        }
    }
    Err("missing '*** End Patch'".into())
}

/// Apply the `@@` chunks of one Update File to `orig`, top to bottom.
fn apply(orig: &str, body: &[&str]) -> Result<String, String> {
    let mut file: Vec<String> = orig.lines().map(str::to_string).collect();
    let mut cursor = 0;
    let mut chunks: Vec<(Option<&str>, Vec<&str>)> = Vec::new();
    for l in body {
        if let Some(h) = l.strip_prefix("@@") {
            chunks.push((Some(h.trim()).filter(|h| !h.is_empty()), Vec::new()));
        } else {
            if chunks.is_empty() {
                chunks.push((None, Vec::new()));
            }
            chunks.last_mut().unwrap().1.push(l);
        }
    }
    for (n, (head, ls)) in chunks.iter().enumerate() {
        let eof = ls.last().is_some_and(|l| l.trim() == "*** End of File");
        let ls = if eof { &ls[..ls.len() - 1] } else { &ls[..] };
        let (mut old, mut new) = (Vec::new(), Vec::new());
        for l in ls {
            match l.chars().next() {
                Some('-') => old.push(&l[1..]),
                Some('+') => new.push(&l[1..]),
                Some(' ') => {
                    old.push(&l[1..]);
                    new.push(&l[1..]);
                }
                None => {
                    old.push("");
                    new.push("");
                }
                _ => return Err(format!("hunk {}: line must start with ' ', '-' or '+': {l:?}", n + 1)),
            }
        }
        if let Some(h) = head {
            // the @@ text names a line (function, class) the hunk sits under
            if let Some(k) = (cursor..file.len()).find(|&k| file[k].trim() == h.trim()).or_else(|| (cursor..file.len()).find(|&k| file[k].contains(h))) {
                cursor = k + 1;
            }
        }
        while old.last() == Some(&"") && new.last() == Some(&"") && !old.is_empty() {
            old.pop();
            new.pop();
        }
        let at = if old.is_empty() {
            if eof || head.is_none() { file.len() } else { cursor }
        } else {
            find(&file, &old, cursor, eof).ok_or_else(|| {
                format!("hunk {}: context not found (after line {}):\n{}", n + 1, cursor, old.join("\n"))
            })?
        };
        file.splice(at..at + old.len(), new.iter().map(|s| s.to_string()));
        cursor = at + new.len();
    }
    let mut out = file.join("\n");
    if !file.is_empty() {
        out.push('\n');
    }
    Ok(out)
}

/// First line index >= `from` where `old` matches: exact, then ignoring trailing, then
/// all surrounding whitespace. `eof` prefers a match that ends the file.
fn find(file: &[String], old: &[&str], from: usize, eof: bool) -> Option<usize> {
    if old.len() > file.len() {
        return None;
    }
    for norm in [(|s: &str| s.to_string()) as fn(&str) -> String, |s| s.trim_end().to_string(), |s| s.trim().to_string()] {
        let hit = |k: usize| old.iter().enumerate().all(|(j, o)| norm(&file[k + j]) == norm(o));
        let last = file.len() - old.len();
        if eof && hit(last) {
            return Some(last);
        }
        if let Some(k) = (from.min(last + 1)..=last).find(|&k| hit(k)).or_else(|| (0..from.min(last + 1)).find(|&k| hit(k))) {
            return Some(k);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::{call, enabled, Config, Ctx};
    use serde_json::json;

    #[test]
    fn v4a_add_update_move_delete() {
        let d = std::env::temp_dir().join(format!("breccia-tools-patch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("a.py"), "def f():\n    return 1\n\n\ndef g():\n    return 2\n").unwrap();
        std::fs::write(d.join("gone.txt"), "x\n").unwrap();
        let mut c = Config::new(d.clone());
        c.allow_write = true;
        let c = Ctx::new(c);
        let t = enabled(&c, None);
        let p = "*** Begin Patch\n*** Update File: a.py\n*** Move to: b.py\n@@ def g():\n-    return 2\n+    return 3\n*** Add File: new/n.txt\n+hello\n*** Delete File: gone.txt\n*** End Patch\n";
        let r = call(&c, &t, "applypatch", &json!({"input": p}));
        assert!(r.contains("R ") && r.contains("A ") && r.contains("D "), "{r}");
        assert_eq!(std::fs::read_to_string(d.join("b.py")).unwrap(), "def f():\n    return 1\n\n\ndef g():\n    return 3\n");
        assert!(!d.join("a.py").exists() && !d.join("gone.txt").exists());
        assert_eq!(std::fs::read_to_string(d.join("new/n.txt")).unwrap(), "hello\n");
        // a failing hunk writes nothing, even for ops before it
        let p = "*** Begin Patch\n*** Add File: x.txt\n+x\n*** Update File: b.py\n@@\n-nope\n+yes\n*** End Patch";
        let r = call(&c, &t, "apply_patch", &json!({"patch": p}));
        assert!(r.contains("context not found"), "{r}");
        assert!(!d.join("x.txt").exists());
        // whitespace-tolerant context, end-of-file append
        let p = "*** Begin Patch\n*** Update File: b.py\n@@\n def f():\n-  return 1\n+    return 0\n@@\n+# end\n*** End of File\n*** End Patch";
        let r = call(&c, &t, "apply_patch", &json!({"patch": p}));
        assert!(r.starts_with("M "), "{r}");
        assert_eq!(std::fs::read_to_string(d.join("b.py")).unwrap(), "def f():\n    return 0\n\n\ndef g():\n    return 3\n# end\n");
        std::fs::remove_dir_all(d).ok();
    }
}
