//! web_search (SearXNG JSON) and web_fetch. HTTP goes through `curl` (no TLS crate).
//! web_fetch resolves the host itself, refuses non-public addresses, pins curl to the
//! checked address (`--resolve`) and follows redirects by hand so every hop is checked.

use std::net::{IpAddr, Ipv4Addr, ToSocketAddrs};
use std::process::Command;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{json, Value};

use crate::{opt_u, s, Args, Ctx, Gate, Tool};

pub fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "web_search",
            description: "Search the web; returns title, URL and snippet per result.",
            schema: || json!({"type": "object", "properties": {
                "query": {"type": "string"}, "limit": {"type": "integer", "description": "Max results (default 8)"}}, "required": ["query"]}),
            gate: Gate::Read,
            keys: &[("q", "query"), ("search_term", "query"), ("search_query", "query"), ("max_results", "limit"), ("num_results", "limit"), ("count", "limit")],
            run: web_search,
        },
        Tool {
            name: "web_fetch",
            description: "Fetch a public http(s) URL and return its text (HTML converted to plain text with [links](url)).",
            schema: || json!({"type": "object", "properties": {
                "url": {"type": "string"}, "max_chars": {"type": "integer", "description": "Default 20000"}}, "required": ["url"]}),
            gate: Gate::Read,
            keys: &[("uri", "url"), ("link", "url"), ("urls", "url"), ("maxChars", "max_chars"), ("max_length", "max_chars"), ("limit", "max_chars")],
            run: web_fetch,
        },
    ]
}

fn curl(args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("curl").args(["-sS", "--proto", "=http,https", "--max-filesize", "5000000"]).args(args).output().map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(out.stdout)
}

fn web_search(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let q = s(a, "query")?;
    let limit = opt_u(a, "limit").unwrap_or(8).clamp(1, 30) as usize;
    let url = format!("{}/search", ctx.cfg.searx.trim_end_matches('/'));
    let body = curl(&["--max-time", "25", "-G", "--data-urlencode", &format!("q={q}"), "--data-urlencode", "format=json", &url])?;
    let v: Value = serde_json::from_slice(&body).map_err(|e| format!("search backend returned non-JSON: {e}"))?;
    let rows: Vec<String> = v["results"]
        .as_array()
        .into_iter()
        .flatten()
        .take(limit)
        .enumerate()
        .map(|(i, r)| {
            let g = |k: &str| r[k].as_str().unwrap_or("").trim().to_string();
            format!("{}. {}\n   {}\n   {}", i + 1, g("title"), g("url"), g("content").chars().take(400).collect::<String>())
        })
        .collect();
    Ok(if rows.is_empty() { format!("no results for {q:?}") } else { rows.join("\n") })
}

/// Addresses web_fetch may not reach: loopback, private, link-local, CGNAT/tailnet
/// (100.64/10), multicast, reserved, ULA, and v6 forms that embed any of those.
pub fn blocked(ip: IpAddr) -> bool {
    fn v4(ip: Ipv4Addr) -> bool {
        let o = ip.octets();
        ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified() || ip.is_broadcast()
            || ip.is_documentation() || ip.is_multicast()
            || o[0] == 0 || o[0] >= 240
            || (o[0] == 100 && (o[1] & 0xC0) == 64)
            || (o[0] == 192 && o[1] == 0 && o[2] == 0)
            || (o[0] == 198 && (o[1] & 0xFE) == 18)
    }
    match ip {
        IpAddr::V4(a) => v4(a),
        IpAddr::V6(a) => {
            let g = a.segments();
            if let Some(m) = a.to_ipv4_mapped() {
                return v4(m);
            }
            let embedded = |hi: u16, lo: u16| Ipv4Addr::new((hi >> 8) as u8, hi as u8, (lo >> 8) as u8, lo as u8);
            a.is_loopback() || a.is_unspecified() || a.is_multicast()
                || (g[0] & 0xFE00) == 0xFC00 // ULA
                || (g[0] & 0xFFC0) == 0xFE80 // link-local
                || (g[0] & 0xFFC0) == 0xFEC0 // site-local
                || (g[0] == 0x2001 && g[1] == 0x0DB8)
                || g[..6].iter().all(|&x| x == 0) // ::/96 (v4-compatible)
                || (g[0] == 0x0064 && g[1] == 0xFF9B && v4(embedded(g[6], g[7]))) // NAT64
                || (g[0] == 0x2002 && v4(embedded(g[1], g[2]))) // 6to4
        }
    }
}

/// (scheme, host, port) of an http(s) URL.
fn parse_url(url: &str) -> Result<(&str, String, u16), String> {
    let (scheme, rest) = url.split_once("://").ok_or("URL needs http:// or https://")?;
    let scheme = match scheme.to_ascii_lowercase().as_str() {
        "http" => "http",
        "https" => "https",
        _ => return Err(format!("only http(s) URLs, not {scheme}:")),
    };
    let auth = rest.split(['/', '?', '#']).next().unwrap_or("");
    let auth = auth.rsplit('@').next().unwrap_or(auth);
    let (host, port) = if let Some(r) = auth.strip_prefix('[') {
        let (h, p) = r.split_once(']').ok_or("bad IPv6 host")?;
        (h.to_string(), p.strip_prefix(':'))
    } else {
        match auth.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p)),
            None => (auth.to_string(), None),
        }
    };
    let port = match port {
        Some(p) if !p.is_empty() => p.parse().map_err(|_| format!("bad port {p:?}"))?,
        _ => if scheme == "https" { 443 } else { 80 },
    };
    if host.is_empty() {
        return Err("URL has no host".into());
    }
    Ok((scheme, host.to_ascii_lowercase(), port))
}

/// A public address for `host:port`, or why not. Every resolved address must be public
/// (a name that also resolves privately is refused).
fn public_addr(host: &str, port: u16) -> Result<IpAddr, String> {
    let addrs: Vec<IpAddr> = (host, port).to_socket_addrs().map_err(|e| format!("cannot resolve {host}: {e}"))?.map(|a| a.ip()).collect();
    match addrs.iter().find(|ip| blocked(**ip)) {
        Some(ip) => Err(format!("{host} resolves to {ip}, a private/local address; web_fetch only reaches public hosts")),
        None => addrs.first().copied().ok_or_else(|| format!("{host}: no addresses")),
    }
}

fn web_fetch(_ctx: &Ctx, a: &Args) -> Result<String, String> {
    let mut url = match a.get("url") {
        Some(Value::Array(v)) => v.first().and_then(Value::as_str).ok_or("url is empty")?.to_string(),
        _ => s(a, "url")?.to_string(),
    };
    let max = opt_u(a, "max_chars").unwrap_or(20_000).clamp(100, 200_000) as usize;
    let body = crate::tmp_path("fetch");
    for _hop in 0..6 {
        let (_, host, port) = parse_url(&url)?;
        let ip = public_addr(&host, port)?;
        let pin = match ip {
            IpAddr::V4(v) => format!("{host}:{port}:{v}"),
            IpAddr::V6(v) => format!("{host}:{port}:[{v}]"),
        };
        let meta = curl(&[
            "--max-time", "30", "--max-redirs", "0", "--resolve", &pin, "-A", "Mozilla/5.0 (compatible; breccia-agent)",
            "-o", body.to_str().unwrap_or("/dev/null"), "-w", "%{http_code}\n%{content_type}\n%{redirect_url}", &url,
        ]);
        let meta = match meta {
            Ok(m) => String::from_utf8_lossy(&m).into_owned(),
            Err(e) => {
                let _ = std::fs::remove_file(&body);
                return Err(e);
            }
        };
        let mut m = meta.lines();
        let code: u16 = m.next().unwrap_or("0").trim().parse().unwrap_or(0);
        let ctype = m.next().unwrap_or("").to_ascii_lowercase();
        let next = m.next().unwrap_or("").trim().to_string();
        if (300..400).contains(&code) && !next.is_empty() {
            url = next;
            continue;
        }
        let bytes = std::fs::read(&body).unwrap_or_default();
        let _ = std::fs::remove_file(&body);
        if !(ctype.is_empty() || ctype.starts_with("text/") || ctype.contains("json") || ctype.contains("xml") || ctype.contains("javascript")) {
            return Err(format!("HTTP {code}: {ctype} is not text ({} bytes)", bytes.len()));
        }
        let raw = String::from_utf8_lossy(&bytes);
        let text = if ctype.contains("html") || (ctype.is_empty() && raw.trim_start().starts_with('<')) { html_to_text(&raw) } else { raw.into_owned() };
        let n = text.chars().count();
        let mut out: String = text.chars().take(max).collect();
        if n > max {
            out.push_str(&format!("\n… [{n} chars, showing {max}; raise max_chars for more]"));
        }
        return Ok(if code >= 400 { format!("HTTP {code}\n{out}") } else { out });
    }
    Err("too many redirects".into())
}

fn res() -> &'static [(Regex, &'static str)] {
    static R: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    R.get_or_init(|| {
        [
            (r"(?is)<!--.*?-->", ""),
            (r"(?is)<script\b.*?</script\s*>", ""),
            (r"(?is)<style\b.*?</style\s*>", ""),
            (r"(?is)<noscript\b.*?</noscript\s*>", ""),
            (r"(?is)<svg\b.*?</svg\s*>", ""),
            (r"(?is)<head\b.*?</head\s*>", ""),
            (r#"(?is)<a\b[^>]*?href\s*=\s*["'](https?://[^"']+)["'][^>]*>(.*?)</a\s*>"#, "[$2]($1)"),
            (r"(?i)<li\b[^>]*>", "\n- "),
            (r"(?i)<(br|/p|p|/div|div|/tr|/h[1-6]|h[1-6]|/li|/ul|/ol|hr|/section|/article|/table|/pre|pre)\b[^>]*>", "\n"),
            (r"(?s)<[^>]*>", ""),
            (r"[ \t\r\f\v]+", " "),
            (r" *\n *", "\n"),
            (r"\n{3,}", "\n\n"),
        ]
        .into_iter()
        .map(|(p, r)| (Regex::new(p).unwrap(), r))
        .collect()
    })
}

pub fn html_to_text(html: &str) -> String {
    static TITLE: OnceLock<Regex> = OnceLock::new();
    let title = TITLE.get_or_init(|| Regex::new(r"(?is)<title[^>]*>(.*?)</title").unwrap()).captures(html).map(|c| c[1].trim().to_string());
    let mut t = html.to_string();
    for (re, rep) in res() {
        t = re.replace_all(&t, *rep).into_owned();
    }
    let t = entities(t.trim());
    match title {
        Some(ti) if !ti.is_empty() => format!("# {}\n\n{t}", entities(&ti)),
        _ => t,
    }
}

fn entities(s: &str) -> String {
    static E: OnceLock<Regex> = OnceLock::new();
    E.get_or_init(|| Regex::new(r"&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z]{2,8});").unwrap())
        .replace_all(s, |c: &regex::Captures| {
            let e = &c[1];
            let ch = if let Some(h) = e.strip_prefix("#x").or_else(|| e.strip_prefix("#X")) {
                u32::from_str_radix(h, 16).ok().and_then(char::from_u32)
            } else if let Some(d) = e.strip_prefix('#') {
                d.parse().ok().and_then(char::from_u32)
            } else {
                match e {
                    "amp" => Some('&'), "lt" => Some('<'), "gt" => Some('>'), "quot" => Some('"'), "apos" => Some('\''),
                    "nbsp" => Some(' '), "mdash" => Some('—'), "ndash" => Some('–'), "hellip" => Some('…'), "copy" => Some('©'),
                    _ => None,
                }
            };
            ch.map_or_else(|| c[0].to_string(), |ch| ch.to_string())
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{call, enabled, Config, Ctx};

    #[test]
    fn ssrf_guard() {
        for ip in ["127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.0.21", "100.64.0.1", "100.127.255.255", "169.254.169.254", "0.0.0.0",
                   "::1", "::ffff:127.0.0.1", "fd7a:115c:a1e0::1", "fe80::1", "64:ff9b::a00:1", "2002:c0a8:1::1", "::7f00:1"] {
            assert!(blocked(ip.parse().unwrap()), "{ip} should be blocked");
        }
        for ip in ["1.1.1.1", "100.128.0.1", "2606:4700:4700::1111", "64:ff9b::101:101"] {
            assert!(!blocked(ip.parse().unwrap()), "{ip} should be allowed");
        }
        assert_eq!(parse_url("https://u:p@Example.com:8443/x?y").unwrap(), ("https", "example.com".into(), 8443));
        assert_eq!(parse_url("http://[::1]/").unwrap(), ("http", "::1".into(), 80));
        assert!(parse_url("file:///etc/passwd").is_err());
        let c = Ctx::new(Config::new(std::env::temp_dir()));
        let t = enabled(&c, None);
        for u in ["http://127.0.0.1:8888/", "http://localhost/", "http://[::ffff:7f00:1]/", "http://100.64.0.5/", "gopher://x/"] {
            let r = call(&c, &t, "fetch", &json!({"url": u}));
            assert!(r.starts_with("error: web_fetch:"), "{u}: {r}");
        }
    }

    #[test]
    fn html_text() {
        let h = "<html><head><title>T &amp; t</title><style>x{}</style></head><body><script>bad()</script><h1>Hi</h1><p>a&nbsp;b &#x41;</p><ul><li>one<li>two</ul><a href=\"https://e.com/x\">link</a></body></html>";
        assert_eq!(html_to_text(h), "# T & t\n\nHi\n\na b A\n\n- one\n- two\n[link](https://e.com/x)");
    }
}
