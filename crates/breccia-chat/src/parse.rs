//! Generated text → reasoning, content and OpenAI `tool_calls`, for every tool-call
//! wire format we know. Formats and examples follow llama.cpp `tests/test-chat.cpp`.

use serde_json::{Map, Value as Json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// gpt-oss: `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{..}<|call|>`
    Harmony,
    /// DeepSeek V3.2: `<｜DSML｜invoke name="N"><｜DSML｜parameter name="k" string="true">v</｜DSML｜parameter>`
    DeepSeekDsml,
    /// DeepSeek V3 / R1 / V3.1: `<｜tool▁call▁begin｜>[function<｜tool▁sep｜>]N…<｜tool▁call▁end｜>`
    DeepSeekV3,
    /// Kimi K2: `<|tool_call_begin|>functions.N:0<|tool_call_argument_begin|>{..}<|tool_call_end|>`
    KimiK2,
    /// MiniMax M2: `<minimax:tool_call><invoke name="N"><parameter name="k">v</parameter></invoke>`
    MiniMax,
    /// GLM 4.5+: `<tool_call>N<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
    Glm,
    /// Qwen3-Coder / Qwen3.5+ / Seed-OSS / Nemotron: `<tool_call><function=N><parameter=k>v</parameter></function></tool_call>`
    QwenXml,
    /// Mistral: `[TOOL_CALLS][{..}]` or `[TOOL_CALLS]N[ARGS]{..}`
    Mistral,
    /// Command R / R7B: `<|START_ACTION|>[{"tool_name": N, "parameters": {..}}]<|END_ACTION|>`
    CommandR,
    /// Granite 3.x: `<|tool_call|>[{"name": N, "arguments": {..}}]`
    Granite,
    /// Hermes 2/3, Qwen 2.5/3, Granite 4, SmolLM3: `<tool_call>{"name": N, "arguments": {..}}</tool_call>`
    HermesJson,
    /// Llama 3.x: `<|python_tag|>{..}` or a bare `{"name": N, "parameters": {..}}`
    Llama3,
    /// Unknown template: try every format but the bare-JSON one.
    Generic,
}

const ORDER: [Format; 12] = [
    Format::QwenXml, Format::HermesJson, Format::Glm, Format::MiniMax, Format::DeepSeekDsml, Format::DeepSeekV3,
    Format::KimiK2, Format::Mistral, Format::CommandR, Format::Granite, Format::Harmony, Format::Llama3,
];

impl Format {
    /// The format a chat template teaches (most specific marker first).
    pub fn detect(src: &str) -> Format {
        let has = |s: &str| src.contains(s);
        if has("<|channel|>") {
            Format::Harmony
        } else if has("DSML") {
            Format::DeepSeekDsml
        } else if has("tool▁calls▁begin") || has("tool▁call▁begin") {
            Format::DeepSeekV3
        } else if has("<|tool_calls_section_begin|>") {
            Format::KimiK2
        } else if has("<minimax:tool_call>") {
            Format::MiniMax
        } else if has("<arg_key>") {
            Format::Glm
        } else if has("<function=") {
            Format::QwenXml
        } else if has("[TOOL_CALLS]") {
            Format::Mistral
        } else if has("<|START_ACTION|>") {
            Format::CommandR
        } else if has("<|tool_call|>") {
            Format::Granite
        } else if has("<tool_call>") {
            Format::HermesJson
        } else if has("<|python_tag|>") || has("\"parameters\": ") && has("ipython") {
            Format::Llama3
        } else {
            Format::Generic
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON object text (OpenAI sends arguments as a string).
    pub arguments: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    pub content: String,
    pub reasoning: Option<String>,
    pub calls: Vec<ToolCall>,
}

const THINK: [(&str, &str); 3] = [("<think>", "</think>"), ("<seed:think>", "</seed:think>"), ("<|START_THINKING|>", "<|END_THINKING|>")];

/// Split `text` (the generated tokens, decoded with their special-token text). `think_open`:
/// the prompt ended inside a reasoning block. `tools`: the request's tool list; calls are
/// only parsed when it is non-empty, and string-typed parameters stay strings.
pub fn parse(format: Format, text: &str, think_open: bool, tools: Option<&Json>) -> Parsed {
    let tools = tools.and_then(|t| t.as_array()).filter(|a| !a.is_empty());
    if format == Format::Harmony {
        return harmony(text, tools.is_some());
    }
    let (reasoning, rest) = split_reasoning(text, think_open);
    let Some(tools) = tools else {
        return Parsed { content: rest.to_string(), reasoning, calls: vec![] };
    };
    let mut tried = vec![format];
    tried.extend(ORDER.iter().copied().filter(|&f| f != format && f != Format::Llama3 && f != Format::Harmony));
    for f in tried {
        if let Some((content, calls)) = calls_in(f, rest, tools).filter(|(_, c)| !c.is_empty()) {
            let calls = calls
                .into_iter()
                .enumerate()
                .map(|(i, (id, name, args))| ToolCall { id: id.unwrap_or_else(|| call_id(i)), name, arguments: args_text(args) })
                .collect();
            return Parsed { content, reasoning, calls };
        }
    }
    Parsed { content: rest.to_string(), reasoning, calls: vec![] }
}

fn split_reasoning(text: &str, think_open: bool) -> (Option<String>, &str) {
    let t = text.trim_start();
    let opened = THINK.iter().find(|(o, _)| t.starts_with(o));
    if !think_open && opened.is_none() {
        return (None, text);
    }
    let body = opened.map_or(text, |(o, _)| &t[o.len()..]);
    match THINK.iter().filter_map(|(_, c)| body.find(c).map(|i| (i, c.len()))).min() {
        Some((i, n)) => (Some(body[..i].trim().to_string()), body[i + n..].trim_start()),
        None => (Some(body.trim().to_string()), ""), // cut off while thinking
    }
}

fn call_id(i: usize) -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_usize(i);
    format!("call_{:016x}", h.finish())
}

fn args_text(v: Json) -> String {
    match v {
        Json::String(s) => s,
        Json::Null => "{}".into(),
        v => v.to_string(),
    }
}

type Call = (Option<String>, String, Json);

fn calls_in(f: Format, text: &str, tools: &[Json]) -> Option<(String, Vec<Call>)> {
    match f {
        Format::QwenXml => qwen_xml(text, tools),
        Format::HermesJson => blocks(text, "<tool_call>", "</tool_call>", &[], |b| json_calls(b)),
        Format::Glm => blocks(text, "<tool_call>", "</tool_call>", &[], |b| glm(b, tools)),
        Format::MiniMax => blocks(text, "<invoke name=", "</invoke>", &[("<minimax:tool_call>", "</minimax:tool_call>")], |b| minimax(b, tools)),
        Format::DeepSeekDsml => blocks(
            text,
            "<｜DSML｜invoke name=",
            "</｜DSML｜invoke>",
            &[("<｜DSML｜function_calls>", "</｜DSML｜function_calls>")],
            |b| dsml(b),
        ),
        Format::DeepSeekV3 => blocks(
            text,
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            &[("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>")],
            |b| deepseek_v3(b),
        ),
        Format::KimiK2 => blocks(
            text,
            "<|tool_call_begin|>",
            "<|tool_call_end|>",
            &[("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>")],
            |b| kimi(b),
        ),
        Format::CommandR => blocks(text, "<|START_ACTION|>", "<|END_ACTION|>", &[], |b| json_calls(b)).map(|(c, k)| {
            (c.replace("<|START_RESPONSE|>", "").replace("<|END_RESPONSE|>", "").trim().to_string(), k)
        }),
        Format::Mistral => tail_json(text, "[TOOL_CALLS]", mistral),
        Format::Granite => tail_json(text, "<|tool_call|>", json_calls),
        Format::Llama3 => {
            let t = text.trim();
            if t.starts_with('{') {
                json_calls(t).map(|c| (String::new(), c))
            } else {
                tail_json(text, "<|python_tag|>", |s| {
                    Some(s.split(';').filter(|p| !p.trim().is_empty()).map(json_calls).collect::<Option<Vec<_>>>()?.concat())
                })
            }
        }
        Format::Harmony | Format::Generic => None,
    }
}

/// Every complete `open … close` block; `inner` gets the text between them (after `open`).
/// Optional wrapper tags around the run of blocks are dropped from the content. An
/// unclosed block (cut off) stays in the content and ends the scan.
fn blocks(
    text: &str,
    open: &str,
    close: &str,
    wrappers: &[(&str, &str)],
    inner: impl Fn(&str) -> Option<Vec<Call>>,
) -> Option<(String, Vec<Call>)> {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(open) {
        let Some(j) = rest[i + open.len()..].find(close) else { break };
        let body = &rest[i + open.len()..i + open.len() + j];
        calls.extend(inner(body)?);
        content.push_str(&rest[..i]);
        rest = &rest[i + open.len() + j + close.len()..];
    }
    if calls.is_empty() {
        return None;
    }
    content.push_str(rest);
    for (o, c) in wrappers {
        content = content.replace(o, "").replace(c, "");
    }
    Some((content.trim().to_string(), calls))
}

/// Content before `marker`, then a parser over everything after it.
fn tail_json(text: &str, marker: &str, f: impl Fn(&str) -> Option<Vec<Call>>) -> Option<(String, Vec<Call>)> {
    let i = text.find(marker)?;
    Some((text[..i].trim().to_string(), f(&text[i + marker.len()..])?))
}

/// One JSON value at the start of `s` (trailing text ignored).
fn first_json(s: &str) -> Option<Json> {
    serde_json::Deserializer::from_str(s.trim_start()).into_iter::<Json>().next()?.ok()
}

/// `{"name"|"tool_name", "arguments"|"parameters"}` or a list of them.
fn json_calls(s: &str) -> Option<Vec<Call>> {
    let one = |v: &Json| -> Option<Call> {
        let o = v.as_object()?;
        let o = o.get("function").and_then(|f| f.as_object()).unwrap_or(o);
        let name = o.get("name").or(o.get("tool_name"))?.as_str()?.to_string();
        let args = o.get("arguments").or(o.get("parameters")).cloned().unwrap_or(Json::Object(Map::new()));
        let id = o.get("id").or(o.get("tool_call_id")).and_then(|i| i.as_str()).map(String::from);
        Some((id.filter(|i| i.len() > 2), name, args))
    };
    match first_json(s)? {
        Json::Array(a) => a.iter().map(one).collect(),
        v => Some(vec![one(&v)?]),
    }
}

/// The schema type(s) of `tool`'s parameter `key`: a string-typed parameter keeps its raw
/// text, anything else is read as JSON when it parses (llama.cpp / vLLM do the same).
fn coerce(tools: &[Json], tool: &str, key: &str, raw: &str) -> Json {
    let prop = tools.iter().find_map(|t| {
        let f = t.get("function").unwrap_or(t);
        (f.get("name")?.as_str()? == tool).then(|| f.get("parameters")?.get("properties")?.get(key)).flatten()
    });
    let types: Vec<&str> = match prop.and_then(|p| p.get("type")) {
        Some(Json::String(s)) => vec![s.as_str()],
        Some(Json::Array(a)) => a.iter().filter_map(|x| x.as_str()).collect(),
        _ => vec![],
    };
    if types.is_empty() || types == ["string"] {
        return Json::String(raw.to_string());
    }
    match serde_json::from_str::<Json>(raw.trim()) {
        Ok(v) if !(v.is_string() && !types.contains(&"string")) || types.contains(&"string") => v,
        _ => Json::String(raw.to_string()),
    }
}

/// `<function=N>` blocks, each optionally wrapped in `<tool_call>` / `<seed:tool_call>`.
fn qwen_xml(text: &str, tools: &[Json]) -> Option<(String, Vec<Call>)> {
    let r = blocks(text, "<function=", "</function>", &[("<tool_call>", "</tool_call>"), ("<seed:tool_call>", "</seed:tool_call>")], |b| {
        let (name, mut rest) = b.split_once('>')?;
        let name = name.trim().to_string();
        let mut args = Map::new();
        while let Some(i) = rest.find("<parameter=") {
            let after = &rest[i + "<parameter=".len()..];
            let (key, after) = after.split_once('>')?;
            let j = after.find("</parameter>")?;
            let v = after[..j].strip_prefix('\n').unwrap_or(&after[..j]);
            let v = v.strip_suffix('\n').unwrap_or(v);
            args.insert(key.trim().to_string(), coerce(tools, &name, key.trim(), v));
            rest = &after[j + "</parameter>".len()..];
        }
        Some(vec![(None, name, Json::Object(args))])
    })?;
    Some(r)
}

/// `N\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>…`
fn glm(b: &str, tools: &[Json]) -> Option<Vec<Call>> {
    let (name, mut rest) = match b.find("<arg_key>") {
        Some(i) => (b[..i].trim(), &b[i..]),
        None => (b.trim(), ""),
    };
    if name.is_empty() || name.starts_with('{') || name.contains('\n') {
        return None; // a Hermes JSON block, not GLM
    }
    let mut args = Map::new();
    while let Some(i) = rest.find("<arg_key>") {
        let (k, after) = rest[i + 9..].split_once("</arg_key>")?;
        let a = after.find("<arg_value>")? + 11;
        let (v, after) = after[a..].split_once("</arg_value>")?;
        args.insert(k.trim().to_string(), coerce(tools, name, k.trim(), v));
        rest = after;
    }
    Some(vec![(None, name.to_string(), Json::Object(args))])
}

/// `"N">\n<parameter name="k">v</parameter>…` (after `<invoke name=`).
fn minimax(b: &str, tools: &[Json]) -> Option<Vec<Call>> {
    let (name, mut rest) = b.split_once('>')?;
    let name = name.trim().trim_matches('"').to_string();
    let mut args = Map::new();
    while let Some(i) = rest.find("<parameter name=") {
        let (k, after) = rest[i + 16..].split_once('>')?;
        let (v, after) = after.split_once("</parameter>")?;
        let k = k.trim().trim_matches('"');
        args.insert(k.to_string(), coerce(tools, &name, k, v.strip_prefix('\n').unwrap_or(v).strip_suffix('\n').unwrap_or(v)));
        rest = after;
    }
    Some(vec![(None, name, Json::Object(args))])
}

/// `"N">\n<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>…`
fn dsml(b: &str) -> Option<Vec<Call>> {
    let (name, mut rest) = b.split_once('>')?;
    let name = name.trim().trim_matches('"').to_string();
    let mut args = Map::new();
    const P: &str = "<｜DSML｜parameter name=";
    while let Some(i) = rest.find(P) {
        let (attrs, after) = rest[i + P.len()..].split_once('>')?;
        let (v, after) = after.split_once("</｜DSML｜parameter>")?;
        let k = attrs.split('"').nth(1)?;
        let raw = !attrs.contains("string=\"false\"");
        let val = if raw { Json::String(v.to_string()) } else { serde_json::from_str(v.trim()).unwrap_or(Json::String(v.to_string())) };
        args.insert(k.to_string(), val);
        rest = after;
    }
    Some(vec![(None, name, Json::Object(args))])
}

/// V3/R1: `function<｜tool▁sep｜>N\n```json\n{..}\n```` ; V3.1: `N<｜tool▁sep｜>{..}`
fn deepseek_v3(b: &str) -> Option<Vec<Call>> {
    let (left, right) = b.split_once("<｜tool▁sep｜>")?;
    if left.trim() == "function" {
        let (name, rest) = right.split_once('\n')?;
        let body = rest.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```");
        return Some(vec![(None, name.trim().to_string(), first_json(body)?)]);
    }
    Some(vec![(None, left.trim().to_string(), first_json(right)?)])
}

/// `functions.N:0<|tool_call_argument_begin|>{..}`; the id is kept (Kimi's template replays it).
fn kimi(b: &str) -> Option<Vec<Call>> {
    let (id, args) = b.split_once("<|tool_call_argument_begin|>")?;
    let id = id.trim();
    let name = id.strip_prefix("functions.").unwrap_or(id);
    let name = name.rsplit_once(':').map_or(name, |(n, _)| n);
    Some(vec![(Some(id.to_string()), name.to_string(), first_json(args)?)])
}

/// After `[TOOL_CALLS]`: a JSON list, or `N[ARGS]{..}` (optionally `N[CALL_ID]id[ARGS]{..}`)
/// with each further call behind its own `[TOOL_CALLS]`.
fn mistral(s: &str) -> Option<Vec<Call>> {
    if s.trim_start().starts_with('[') {
        return json_calls(s);
    }
    s.split("[TOOL_CALLS]")
        .map(|part| {
            let (head, args) = part.split_once("[ARGS]")?;
            let (name, id) = match head.split_once("[CALL_ID]") {
                Some((n, i)) => (n, Some(i.trim().to_string())),
                None => (head, None),
            };
            Some((id, name.trim().to_string(), first_json(args)?))
        })
        .collect()
}

/// gpt-oss: messages split by `<|start|>assistant`; `analysis` = reasoning,
/// `final` / `commentary` = content, a `to=functions.N` recipient = a call.
fn harmony(text: &str, tools: bool) -> Parsed {
    let mut p = Parsed::default();
    let mut reasoning = Vec::new();
    for msg in text.split("<|start|>assistant") {
        let Some((header, body)) = msg.split_once("<|message|>") else {
            continue;
        };
        let end = ["<|end|>", "<|call|>", "<|return|>"].iter().filter_map(|e| body.find(e)).min().unwrap_or(body.len());
        let body = &body[..end];
        let channel = header.split("<|channel|>").nth(1).and_then(|c| c.split_whitespace().next()).unwrap_or("");
        match header.split_once("to=functions.") {
            Some((_, n)) if tools => {
                let name: String = n.chars().take_while(|c| !c.is_whitespace() && *c != '<').collect();
                if let Some(args) = first_json(body) {
                    p.calls.push(ToolCall { id: call_id(p.calls.len()), name, arguments: args_text(args) });
                }
            }
            _ if channel == "analysis" => reasoning.push(body.trim().to_string()),
            _ => p.content.push_str(body),
        }
    }
    p.reasoning = (!reasoning.is_empty()).then(|| reasoning.join("\n"));
    p.content = p.content.trim().to_string();
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Json {
        json!([
            {"type": "function", "function": {"name": "special_function", "parameters": {"type": "object", "properties": {"arg1": {"type": "integer"}}}}},
            {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
        ])
    }

    fn one(f: Format, text: &str, think: bool) -> Parsed {
        let p = parse(f, text, think, Some(&tools()));
        assert_eq!(p.calls.len(), 1, "{f:?}: {p:?}");
        p
    }

    fn check(f: Format, text: &str, name: &str, args: &str) {
        let p = one(f, text, false);
        assert_eq!((p.calls[0].name.as_str(), p.calls[0].arguments.as_str()), (name, args), "{f:?}");
        assert!(p.calls[0].id.starts_with("call_") || p.calls[0].id.starts_with("functions."));
    }

    #[test]
    fn qwen_xml_with_reasoning_and_content() {
        let t = "I should check.\n</think>\n\nLet me look.\n\n<tool_call>\n<function=get_time>\n<parameter=city>\n12 Main\nSt\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>";
        let p = parse(Format::QwenXml, t, true, Some(&tools()));
        assert_eq!(p.reasoning.as_deref(), Some("I should check."));
        assert_eq!(p.content, "Let me look.");
        assert_eq!(p.calls[0].arguments, r#"{"city":"12 Main\nSt"}"#);
        assert_eq!(p.calls[1].arguments, r#"{"arg1":1}"#);
        // seed-oss wrapper, no newlines
        check(Format::Generic, "<seed:tool_call>\n<function=special_function>\n<parameter=arg1>1</parameter>\n</function>\n</seed:tool_call>", "special_function", r#"{"arg1":1}"#);
    }

    #[test]
    fn cut_off_call_stays_content() {
        let t = "</think>ok<tool_call>\n<function=get_time>\n<parameter=city>\nPar";
        let p = parse(Format::QwenXml, t, true, Some(&tools()));
        assert!(p.calls.is_empty());
        assert!(p.content.starts_with("ok<tool_call>"));
        let p = parse(Format::QwenXml, "still thinking", true, Some(&tools()));
        assert_eq!((p.reasoning.as_deref(), p.content.as_str()), (Some("still thinking"), ""));
        let p = parse(Format::QwenXml, "plain answer", false, None);
        assert_eq!((p.reasoning, p.content.as_str()), (None, "plain answer"));
    }

    #[test]
    fn every_format() {
        check(Format::HermesJson, "<tool_call>\n{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n</tool_call>", "special_function", r#"{"arg1":1}"#);
        check(Format::Glm, "<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>", "special_function", r#"{"arg1":1}"#);
        check(Format::Glm, "<tool_call>get_time<arg_key>city</arg_key><arg_value>42</arg_value></tool_call>", "get_time", r#"{"city":"42"}"#);
        check(Format::MiniMax, "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>", "special_function", r#"{"arg1":1}"#);
        check(Format::DeepSeekDsml, "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_time\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>", "get_time", r#"{"city":"Tokyo"}"#);
        check(Format::DeepSeekDsml, "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"special_function\">\n<｜DSML｜parameter name=\"arg1\" string=\"false\">1</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>", "special_function", r#"{"arg1":1}"#);
        check(Format::DeepSeekV3, "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"XYZ\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>", "get_time", r#"{"city":"XYZ"}"#);
        check(Format::DeepSeekV3, "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n```json\n{\"arg1\": 1}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>", "special_function", r#"{"arg1":1}"#);
        let k = one(Format::KimiK2, "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>", false);
        assert_eq!((k.calls[0].id.as_str(), k.calls[0].name.as_str()), ("functions.special_function:0", "special_function"));
        check(Format::Mistral, r#"[TOOL_CALLS]special_function[ARGS]{"arg1":1}"#, "special_function", r#"{"arg1":1}"#);
        check(Format::Mistral, r#"[TOOL_CALLS][{"name": "special_function", "arguments": {"arg1": 1}}]"#, "special_function", r#"{"arg1":1}"#);
        let two = parse(Format::Mistral, r#"[TOOL_CALLS]special_function[ARGS]{"arg1": 1}[TOOL_CALLS]get_time[ARGS]{"city": "a"}"#, false, Some(&tools()));
        assert_eq!(two.calls.len(), 2);
        let c = parse(Format::CommandR, "<|START_THINKING|>I'm\nthinking<|END_THINKING|><|START_ACTION|>[\n    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n]<|END_ACTION|>", false, Some(&tools()));
        assert_eq!((c.reasoning.as_deref(), c.calls[0].arguments.as_str()), (Some("I'm\nthinking"), r#"{"arg1":1}"#));
        check(Format::Granite, r#"<|tool_call|>[{"name": "special_function", "arguments": {"arg1": 1}}]"#, "special_function", r#"{"arg1":1}"#);
        check(Format::Llama3, r#"{"name": "special_function", "parameters": {"arg1": 1}}"#, "special_function", r#"{"arg1":1}"#);
        check(Format::Llama3, r#"<|python_tag|>{"name": "special_function", "parameters": {"arg1": 1}}"#, "special_function", r#"{"arg1":1}"#);
        // the bare-JSON reading is never a fallback: a JSON answer stays content
        let p = parse(Format::QwenXml, r#"{"name": "x", "parameters": {}}"#, false, Some(&tools()));
        assert!(p.calls.is_empty());
    }

    #[test]
    fn harmony() {
        let p = parse(Format::Harmony, "<|channel|>analysis<|message|>I'm\nthinking<|end|><|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}<|call|>", false, Some(&tools()));
        assert_eq!(p.reasoning.as_deref(), Some("I'm\nthinking"));
        assert_eq!((p.calls[0].name.as_str(), p.calls[0].arguments.as_str()), ("special_function", r#"{"arg1":1}"#));
        let p = parse(Format::Harmony, " to=functions.special_function<|channel|>analysis<|message|>{\"arg1\": 1}", false, Some(&tools()));
        assert_eq!(p.calls.len(), 1);
        let p = parse(Format::Harmony, "<|channel|>analysis<|message|>hm<|end|><|start|>assistant<|channel|>final<|message|>Hello", false, None);
        assert_eq!((p.reasoning.as_deref(), p.content.as_str()), (Some("hm"), "Hello"));
    }

    #[test]
    fn detect() {
        assert_eq!(Format::detect("{{'<tool_call>\\n<function=' + n}}"), Format::QwenXml);
        assert_eq!(Format::detect("<tool_call>{{ x|tojson }}</tool_call>"), Format::HermesJson);
        assert_eq!(Format::detect("{{ x }}"), Format::Generic);
    }
}
