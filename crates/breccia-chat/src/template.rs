//! Hugging Face-compatible Jinja rendering (minijinja).

use std::fmt::Write as _;

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::{Map, Value as Json};

use crate::parse::Format;

/// A compiled chat template plus what the output parser needs to know about it.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
    /// The tool-call wire format the template teaches the model.
    pub format: Format,
}

/// A rendered prompt.
#[derive(Debug)]
pub struct Rendered {
    pub text: String,
    /// The generation prompt opened a reasoning block (`<think>` and the like), so the
    /// output starts inside it.
    pub think_open: bool,
}

impl ChatTemplate {
    /// Compile `source` (a Jinja chat template). `bos` / `eos` are the token texts the
    /// template may reference as `bos_token` / `eos_token`.
    pub fn new(source: &str, bos: &str, eos: &str) -> Result<Self, String> {
        let mut env = Environment::new();
        // HF renders with trim_blocks + lstrip_blocks and no autoescape
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| e.to_string())?,
        );
        env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_filter("indent", indent);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| strftime_now(&fmt));
        // compiled once per process
        let src: &'static str = Box::leak(rewrite_mutations(source).into_boxed_str());
        env.add_template("chat", src).map_err(|e| format!("chat template: {e:#}"))?;
        Ok(Self { env, bos: bos.into(), eos: eos.into(), format: Format::detect(source) })
    }

    /// Render an OpenAI chat request. Top-level `enable_thinking`, `reasoning_effort`
    /// and `preserve_thinking` and every `chat_template_kwargs` entry reach the template.
    pub fn render(&self, req: &Json) -> Result<Rendered, String> {
        let msgs = req.get("messages").and_then(|m| m.as_array()).ok_or("messages must be an array")?;
        let mut ctx = Map::new();
        ctx.insert("messages".into(), Json::Array(msgs.iter().map(normalize).collect()));
        if let Some(t) = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
            ctx.insert("tools".into(), t.clone());
        }
        ctx.insert("add_generation_prompt".into(), Json::Bool(true));
        ctx.insert("bos_token".into(), Json::String(self.bos.clone()));
        ctx.insert("eos_token".into(), Json::String(self.eos.clone()));
        for k in ["enable_thinking", "reasoning_effort", "preserve_thinking"] {
            if let Some(v) = req.get(k).filter(|v| !v.is_null()) {
                ctx.insert(k.into(), v.clone());
            }
        }
        if let Some(Json::Object(kw)) = req.get("chat_template_kwargs") {
            ctx.extend(kw.clone());
        }
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        let text = tmpl.render(Value::from(minijinja::value::Serde(&ctx))).map_err(|e| format!("chat template: {e}"))?;
        let tail = text.trim_end();
        let think_open = ["<think>", "<seed:think>", "<|START_THINKING|>"].iter().any(|t| tail.ends_with(t));
        Ok(Rendered { text, think_open })
    }
}

/// OpenAI sends `tool_calls[].function.arguments` as a JSON string; templates iterate it
/// as a mapping (HF requires parsing it first). Unparseable strings are left for the
/// template to reject.
fn normalize(m: &Json) -> Json {
    let mut m = m.clone();
    if let Some(Json::Array(calls)) = m.get_mut("tool_calls") {
        for c in calls {
            if let Some(f) = c.get_mut("function") {
                if let Some(Json::String(s)) = f.get("arguments") {
                    if let Ok(v @ Json::Object(_)) = serde_json::from_str::<Json>(s) {
                        f["arguments"] = v;
                    } else if s.trim().is_empty() {
                        f["arguments"] = Json::Object(Map::new());
                    }
                }
            }
        }
    }
    m
}

/// Python `json.dumps(v, ensure_ascii=False, indent=indent)`: `", "` / `": "` separators
/// (`","` with an indent), insertion order, no HTML escaping (minijinja's own `tojson`
/// escapes `<>&'`, which changes the prompt).
fn tojson(v: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let mut j = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    if sort == Some(true) {
        sort_keys(&mut j);
    }
    let mut out = String::new();
    dump(&j, indent, 0, &mut out);
    if ascii == Some(true) {
        // non-ASCII only occurs inside JSON strings, so escaping the whole text is exact
        out = out.chars().fold(String::new(), |mut a, c| {
            if c.is_ascii() {
                a.push(c);
            } else {
                for u in c.encode_utf16(&mut [0; 2]) {
                    a.push_str(&format!("\\u{u:04x}"));
                }
            }
            a
        });
    }
    Ok(Value::from_safe_string(out))
}

/// jinja2's `indent` (minijinja's drops a trailing newline, which Command R relies on).
fn indent(s: String, width: Option<Value>, first: Option<bool>, blank: Option<bool>, kw: Kwargs) -> Result<String, Error> {
    let width = match width { Some(w) => Some(w), None => kw.get("width")? };
    let first = match first { Some(f) => f, None => kw.get::<Option<bool>>("first")?.unwrap_or(false) };
    let blank = match blank { Some(b) => b, None => kw.get::<Option<bool>>("blank")?.unwrap_or(false) };
    kw.assert_all_used()?;
    let pad = match width {
        Some(w) if w.as_str().is_some() => w.as_str().unwrap().to_string(),
        Some(w) => " ".repeat(usize::try_from(w)?),
        None => "    ".into(),
    };
    let s = s + "\n"; // jinja2 appends one, then splitlines() eats it
    let lines: Vec<&str> = s.lines().collect();
    let mut rv = if blank {
        lines.join(&format!("\n{pad}"))
    } else {
        let rest: Vec<String> = lines[1..].iter().map(|l| if l.is_empty() { String::new() } else { format!("{pad}{l}") }).collect();
        if rest.is_empty() { lines[0].to_string() } else { format!("{}\n{}", lines[0], rest.join("\n")) }
    };
    if first {
        rv.insert_str(0, &pad);
    }
    Ok(rv)
}

fn sort_keys(v: &mut Json) {
    match v {
        Json::Object(m) => {
            m.sort_keys();
            m.values_mut().for_each(sort_keys);
        }
        Json::Array(a) => a.iter_mut().for_each(sort_keys),
        _ => {}
    }
}

/// minijinja lists are immutable. Kimi K2 keeps a queue in a namespace with
/// `{% set _ = ns.ids.append(x) %}` / `{% set id = ns.ids.pop(0) %}`; rewrite those tags to
/// rebind the attribute (`ns.ids = ns.ids + [x]`, `id = ns.ids[0]` then `ns.ids = ns.ids[1:]`).
/// shortcut: only `set NAME = PATH.append(..)|.pop(..)` tags, extend if a template mutates otherwise.
fn rewrite_mutations(src: &str) -> String {
    let mut out = String::new();
    let mut rest = src;
    while let Some(i) = rest.find("{%") {
        out.push_str(&rest[..i]);
        let Some(j) = rest[i..].find("%}") else { break };
        let tag = &rest[i..i + j + 2];
        rest = &rest[i + j + 2..];
        out.push_str(&mutation(tag).unwrap_or_else(|| tag.to_string()));
    }
    out.push_str(rest);
    out
}

fn mutation(tag: &str) -> Option<String> {
    let open = if tag.starts_with("{%-") { "{%-" } else { "{%" };
    let close = if tag.ends_with("-%}") { "-%}" } else { "%}" };
    let body = tag.strip_prefix(open)?.strip_suffix(close)?.trim();
    let (var, call) = body.strip_prefix("set ")?.split_once('=')?;
    let (var, call) = (var.trim(), call.trim());
    let (path, args) = call.strip_suffix(')')?.split_once('(')?;
    if let Some(path) = path.strip_suffix(".append") {
        return Some(format!("{open} set {path} = ({path} + [{args}]) | list {close}"));
    }
    let path = path.strip_suffix(".pop")?;
    let (at, keep) = match args.trim() {
        "" | "-1" => ("-1", "[:-1]"),
        "0" => ("0", "[1:]"),
        _ => return None,
    };
    Some(format!("{open} set {var} = {path}[{at}] {close}{open} set {path} = {path}{keep} | list {close}"))
}

pub(crate) fn dump(v: &Json, indent: Option<usize>, depth: usize, out: &mut String) {
    let nl = |out: &mut String, d: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', n * d));
        }
    };
    let sep = if indent.is_some() { "," } else { ", " };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => match n.as_f64().filter(|_| n.is_f64()) {
            Some(f) if f.is_finite() && f == f.trunc() && f.abs() < 1e16 => {
                let _ = write!(out, "{f:.1}"); // Python repr(1.0) == "1.0"
            }
            _ => out.push_str(&n.to_string()),
        },
        Json::String(s) => py_str(s, out),
        Json::Array(a) if a.is_empty() => out.push_str("[]"),
        Json::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep);
                }
                nl(out, depth + 1);
                dump(x, indent, depth + 1, out);
            }
            nl(out, depth);
            out.push(']');
        }
        Json::Object(o) if o.is_empty() => out.push_str("{}"),
        Json::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep);
                }
                nl(out, depth + 1);
                py_str(k, out);
                out.push_str(": ");
                dump(x, indent, depth + 1, out);
            }
            nl(out, depth);
            out.push('}');
        }
    }
}

fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `strftime_now(fmt)` (Llama 3 and others date their system prompt), UTC.
fn strftime_now(fmt: &str) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    let month = MONTHS[(m - 1) as usize];
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let _ = match it.next() {
            Some('Y') => write!(out, "{y}"),
            Some('m') => write!(out, "{m:02}"),
            Some('d') => write!(out, "{d:02}"),
            Some('B') => write!(out, "{month}"),
            Some('b') => write!(out, "{}", &month[..3]),
            Some('H') => write!(out, "{:02}", rem / 3600),
            Some('M') => write!(out, "{:02}", rem / 60 % 60),
            Some('S') => write!(out, "{:02}", rem % 60),
            Some('%') => write!(out, "%"),
            Some(o) => write!(out, "%{o}"),
            None => write!(out, "%"),
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tojson_matches_python() {
        let mut s = String::new();
        dump(&json!({"b": 1, "a": [1.0, "x<y>\n\"é"], "e": {}}), None, 0, &mut s);
        assert_eq!(s, r#"{"b": 1, "a": [1.0, "x<y>\n\"é"], "e": {}}"#);
        let mut s = String::new();
        dump(&json!({"a": [1, 2]}), Some(2), 0, &mut s);
        assert_eq!(s, "{\n  \"a\": [\n    1,\n    2\n  ]\n}");
        let t = ChatTemplate::new("{{ tools[0] | tojson(ensure_ascii=True, sort_keys=True) }}", "", "").unwrap();
        let r = t.render(&json!({"messages": [], "tools": [{"z": "é😀", "a": {"y": 1, "b": 2}}]})).unwrap();
        assert_eq!(r.text, r#"{"a": {"b": 2, "y": 1}, "z": "\u00e9\ud83d\ude00"}"#);
    }

    #[test]
    fn indent_like_jinja2() {
        let t = ChatTemplate::new("{{ '{\\n'|indent(4, first=true) }}|{{ 'a\\n\\nb'|indent(2) }}|{{ 'x\\ny'|indent(width='> ', blank=true) }}", "", "").unwrap();
        assert_eq!(t.render(&json!({"messages": []})).unwrap().text, "    {\n|a\n\n  b|x\n> y");
    }

    #[test]
    fn append_rewrite() {
        let t = ChatTemplate::new(
            "{%- set q = namespace(ids=[]) -%}{%- for m in messages -%}{%- set _ = q.ids.append('f(' ~ m.role ~ ')') -%}{%- endfor -%}{% set h = q.ids.pop(0) %}{{ h }}:{{ q.ids | join(',') }} set x",
            "", "",
        )
        .unwrap();
        assert_eq!(t.render(&json!({"messages": [{"role": "a"}, {"role": "b"}]})).unwrap().text, "f(a):f(b) set x");
    }

    #[test]
    fn date_has_shape() {
        let d = strftime_now("%d %b %Y");
        assert_eq!(d.len(), 11, "{d}");
    }

    #[test]
    fn hf_features_render() {
        let t = ChatTemplate::new(
            "{%- for m in messages[::-1] %}{{ m.content.strip() }}{% if not loop.last %}|{{ loop.nextitem.role }}|{% endif %}{%- endfor %}\n{{ tools[0] | tojson }}{{ raise_exception('x') if false }}",
            "", "",
        )
        .unwrap();
        let r = t.render(&json!({"messages": [{"role": "user", "content": " a "}, {"role": "assistant", "content": "b"}], "tools": [{"k": "<v>"}]})).unwrap();
        assert_eq!(r.text, "b|user|a{\"k\": \"<v>\"}", "trim_blocks eats the newline after endfor, as jinja2 does");
        let e = ChatTemplate::new("{{ raise_exception('bad ' ~ messages|length) }}", "", "").unwrap();
        assert!(e.render(&json!({"messages": []})).unwrap_err().contains("bad 0"));
    }
}
