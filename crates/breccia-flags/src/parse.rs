//! Parsing of llama.cpp-compatible `llama-server` argument vectors.

use std::path::PathBuf;

use crate::device::{Backend, DeviceRef};
use crate::error::FlagError;
use crate::overrides::parse_override_list;
use crate::plan::LaunchPlan;
use crate::value::{KvCacheType, ReasoningMode, Switch};

/// Parse a `llama-server` argv (flags only) into a [`LaunchPlan`].
///
/// `argv` is the argument vector **without** being required to strip the program
/// name: a leading token that does not start with `-` and looks like a program
/// path (`/usr/bin/llama-server`, `llama-server`, `...exe`) is accepted as the
/// program name and recorded in [`LaunchPlan::binary`] unless `-D` overrides it.
///
/// Accepted forms for every flag:
///
/// - `-c 200000` — space-separated value (the value token is taken verbatim, so
///   `-ngl -1` works);
/// - `--port=8160` / `-c=200000` — inline `=value`;
/// - boolean flags (`--jinja`) take no value, but tolerate `--jinja=false`.
///
/// Anything else — an unknown flag, a bare positional, a duplicate flag, or a
/// value that fails to parse — is a [`FlagError`] naming the flag and its 0-based
/// argument position. Nothing in this module panics on malformed input.
pub fn parse_llama_server_args(argv: &[String]) -> Result<LaunchPlan, FlagError> {
    let mut plan = LaunchPlan::default();
    let mut seen = Seen::default();
    let mut index = 0usize;

    if let Some(first) = argv.first() {
        if !first.starts_with('-') && looks_like_program(first) {
            plan.binary = Some(PathBuf::from(first));
            index = 1;
        }
    }

    while index < argv.len() {
        let position = index;
        let raw = argv[index].as_str();

        if raw == "--" {
            return match argv.get(index + 1) {
                Some(extra) => Err(FlagError::UnexpectedPositional {
                    value: extra.clone(),
                    position: index + 1,
                }),
                None => Ok(plan),
            };
        }
        if raw.is_empty() || !raw.starts_with('-') || raw == "-" {
            return Err(FlagError::UnexpectedPositional {
                value: raw.to_string(),
                position,
            });
        }

        let (name, inline) = split_flag(raw);
        let mut frame = Frame {
            argv,
            position,
            name,
            inline,
            consumed: position + 1,
        };
        // Position of the next unread token; arms that consume a value overwrite it.
        let mut next = position + 1;

        match name {
            // ---- binary / model ------------------------------------------------
            "-D" | "--binary" => {
                seen.mark("binary", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.binary = Some(PathBuf::from(value));
                next = consumed;
            }
            "-m" | "--model" => {
                seen.mark("model", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.model = Some(PathBuf::from(value));
                next = consumed;
            }

            // ---- backend / devices ---------------------------------------------
            "--backend" | "-be" => {
                seen.mark("backend", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.backend =
                    value
                        .parse::<Backend>()
                        .map_err(|reason| FlagError::InvalidValue {
                            flag: name.to_string(),
                            value: value.clone(),
                            position,
                            reason,
                        })?;
                next = consumed;
            }
            "--device" | "--devices" | "-dev" => {
                let (value, consumed) = frame.value()?;
                for part in value.split(',') {
                    let alias = part.trim();
                    if alias.is_empty() {
                        return Err(FlagError::InvalidDevice {
                            value: part.to_string(),
                            position,
                            reason: "empty device reference (stray comma?)".to_string(),
                        });
                    }
                    let device =
                        DeviceRef::parse(alias).map_err(|reason| FlagError::InvalidDevice {
                            value: alias.to_string(),
                            position,
                            reason,
                        })?;
                    plan.devices.push(device);
                }
                next = consumed;
            }

            // ---- placement ------------------------------------------------------
            "-ngl" | "--n-gpu-layers" | "--gpu-layers" => {
                seen.mark("gpu_layers", name, position)?;
                plan.gpu_layers = Some(frame.num::<i32>()?);
                next = frame.consumed;
            }
            "-ts" | "--tensor-split" => {
                seen.mark("tensor_split", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.tensor_split = Some(parse_f32_list(name, &value, position)?);
                next = consumed;
            }
            "-ot" | "--override-tensor" => {
                let (value, consumed) = frame.value()?;
                let base = plan.override_tensors.len();
                plan.override_tensors
                    .extend(parse_override_list(&value, base)?);
                next = consumed;
            }
            "-cmoe" | "--cpu-moe" => {
                seen.mark("cpu_moe", name, position)?;
                plan.cpu_moe = frame.bool_flag()?;
            }
            "--moe-cache-mib" => {
                seen.mark("moe_cache_mib", name, position)?;
                plan.moe_cache_mib = Some(frame.num::<u64>()?);
                next = frame.consumed;
            }

            // ---- fork-specific switches -----------------------------------------
            "-lzm" => {
                seen.mark("lzm", name, position)?;
                let (value, consumed) = frame.switch()?;
                plan.lzm = Some(value);
                next = consumed;
            }
            "-lm" => {
                seen.mark("lm", name, position)?;
                let (value, consumed) = frame.value()?;
                if value.trim().is_empty() {
                    return Err(FlagError::InvalidValue {
                        flag: name.to_string(),
                        value,
                        position,
                        reason: "expected a mode name such as `none`".to_string(),
                    });
                }
                plan.lm = Some(value);
                next = consumed;
            }
            "--no-host" => {
                seen.mark("no_host", name, position)?;
                plan.no_host = frame.bool_flag()?;
            }
            "--host-expert-import" => {
                seen.mark("host_expert_import", name, position)?;
                plan.host_expert_import = frame.bool_flag()?;
            }

            // ---- context / KV ---------------------------------------------------
            "-c" | "--ctx-size" => {
                seen.mark("ctx_size", name, position)?;
                plan.ctx_size = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }
            "-kvu" | "--kv-unified" => {
                seen.mark("kv_unified", name, position)?;
                plan.kv_unified = frame.bool_flag()?;
            }
            "-np" | "--parallel" => {
                seen.mark("parallel", name, position)?;
                plan.parallel = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }
            "--kv-unified-per-slot" => {
                seen.mark("kv_unified_per_slot", name, position)?;
                plan.kv_unified_per_slot = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }
            "-ctk" | "--cache-type-k" => {
                seen.mark("cache_type_k", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.cache_type_k = Some(KvCacheType::parse_value(&value).map_err(|reason| {
                    FlagError::InvalidValue {
                        flag: name.to_string(),
                        value: value.clone(),
                        position,
                        reason,
                    }
                })?);
                next = consumed;
            }
            "-ctv" | "--cache-type-v" => {
                seen.mark("cache_type_v", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.cache_type_v = Some(KvCacheType::parse_value(&value).map_err(|reason| {
                    FlagError::InvalidValue {
                        flag: name.to_string(),
                        value: value.clone(),
                        position,
                        reason,
                    }
                })?);
                next = consumed;
            }
            "-fa" | "--flash-attn" => {
                seen.mark("flash_attn", name, position)?;
                let (value, consumed) = frame.switch()?;
                plan.flash_attn = Some(value);
                next = consumed;
            }
            "-cram" => {
                seen.mark("cram", name, position)?;
                plan.cram = Some(frame.num::<u64>()?);
                next = frame.consumed;
            }
            "-ctxcp" => {
                seen.mark("ctxcp", name, position)?;
                plan.ctxcp = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }
            "--slot-save-path" => {
                seen.mark("slot_save_path", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.slot_save_path = Some(PathBuf::from(value));
                next = consumed;
            }

            // ---- multimodal -----------------------------------------------------
            "--mmproj" => {
                seen.mark("mmproj", name, position)?;
                let (value, consumed) = frame.value()?;
                plan.mmproj = Some(PathBuf::from(value));
                next = consumed;
            }
            "--mmdev" | "-mmdev" | "--mmproj-device" => {
                seen.mark("mmproj_device", name, position)?;
                let (value, consumed) = frame.value()?;
                let alias = value.trim();
                plan.mmproj_device =
                    Some(
                        DeviceRef::parse(alias).map_err(|reason| FlagError::InvalidDevice {
                            value: alias.to_string(),
                            position,
                            reason,
                        })?,
                    );
                next = consumed;
            }
            "--image-min-tokens" => {
                seen.mark("image_min_tokens", name, position)?;
                plan.image_min_tokens = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }
            "--image-max-tokens" => {
                seen.mark("image_max_tokens", name, position)?;
                plan.image_max_tokens = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }

            // ---- logging / threads ----------------------------------------------
            "-lv" | "--log-verbosity" => {
                seen.mark("log_verbosity", name, position)?;
                plan.log_verbosity = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }
            "--reasoning" => {
                seen.mark("reasoning", name, position)?;
                let (value, consumed) = frame.reasoning()?;
                plan.reasoning = Some(value);
                next = consumed;
            }
            "-t" | "--threads" => {
                seen.mark("threads", name, position)?;
                plan.threads = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }
            "-tb" | "--threads-batch" => {
                seen.mark("threads_batch", name, position)?;
                plan.threads_batch = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }

            // ---- server ---------------------------------------------------------
            "--host" => {
                seen.mark("host", name, position)?;
                let (value, consumed) = frame.value()?;
                if value.trim().is_empty() {
                    return Err(FlagError::InvalidValue {
                        flag: name.to_string(),
                        value,
                        position,
                        reason: "expected an address such as 127.0.0.1".to_string(),
                    });
                }
                plan.host = Some(value);
                next = consumed;
            }
            "--port" => {
                seen.mark("port", name, position)?;
                plan.port = Some(frame.num::<u16>()?);
                next = frame.consumed;
            }
            "--jinja" => {
                seen.mark("jinja", name, position)?;
                plan.jinja = frame.bool_flag()?;
            }
            "--metrics" => {
                seen.mark("metrics", name, position)?;
                plan.metrics = frame.bool_flag()?;
            }
            "--fit" => {
                seen.mark("fit", name, position)?;
                let (value, consumed) = frame.switch()?;
                plan.fit = Some(value);
                next = consumed;
            }
            "--no-op-offload" => {
                seen.mark("no_op_offload", name, position)?;
                plan.no_op_offload = frame.bool_flag()?;
            }
            "-ub" | "--ubatch-size" => {
                seen.mark("ubatch_size", name, position)?;
                plan.ubatch_size = Some(frame.num::<u32>()?);
                next = frame.consumed;
            }

            other => {
                return Err(FlagError::UnknownFlag {
                    flag: other.to_string(),
                    position,
                })
            }
        }

        index = next;
    }

    Ok(plan)
}

/// Parse a whitespace-separated launch line (quotes allowed) into a [`LaunchPlan`].
///
/// Convenience entry point for tests, config files and `breccia-cli` dry runs —
/// a real launch should hand over an argv slice instead. Rules:
///
/// - whitespace separates tokens;
/// - `'...'` and `"..."` group a token and are removed (so a quoted `-ot`
///   pattern reaches the parser intact);
/// - a backslash followed by a newline is a line continuation; every other
///   backslash is **literal**, so regex escapes survive unquoted;
/// - no other escapes are interpreted.
pub fn parse_str(input: &str) -> Result<LaunchPlan, FlagError> {
    let argv = tokenize(input)?;
    parse_llama_server_args(&argv)
}

/// Split a launch line into argv tokens, honouring quotes and line continuations.
///
/// See [`parse_str`] for the rules. Returns [`FlagError::Tokenize`] for an
/// unterminated quote.
pub fn tokenize(input: &str) -> Result<Vec<String>, FlagError> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut has_token = false;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            c if c.is_whitespace() => {
                if has_token {
                    tokens.push(std::mem::take(&mut current));
                    has_token = false;
                }
            }
            '"' | '\'' => {
                has_token = true;
                let quote = ch;
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == quote {
                        closed = true;
                        break;
                    }
                    current.push(inner);
                }
                if !closed {
                    return Err(FlagError::Tokenize {
                        position: tokens.len(),
                        reason: format!("unterminated {quote} quote"),
                    });
                }
            }
            '\\' if chars.peek() == Some(&'\n') => {
                chars.next();
            }
            other => {
                has_token = true;
                current.push(other);
            }
        }
    }

    if has_token {
        tokens.push(current);
    }
    Ok(tokens)
}

/// One flag occurrence being parsed.
struct Frame<'a> {
    argv: &'a [String],
    position: usize,
    name: &'a str,
    inline: Option<&'a str>,
    /// Set by the value helpers to the index of the next unread token.
    consumed: usize,
}

impl Frame<'_> {
    fn value(&self) -> Result<(String, usize), FlagError> {
        match self.inline {
            Some(value) if !value.is_empty() => Ok((value.to_string(), self.position + 1)),
            Some(_) => Err(FlagError::MissingValue {
                flag: self.name.to_string(),
                position: self.position,
            }),
            None => match self.argv.get(self.position + 1) {
                Some(value) => Ok((value.clone(), self.position + 2)),
                None => Err(FlagError::MissingValue {
                    flag: self.name.to_string(),
                    position: self.position,
                }),
            },
        }
    }

    fn num<T>(&mut self) -> Result<T, FlagError>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        let (value, consumed) = self.value()?;
        self.consumed = consumed;
        value.parse::<T>().map_err(|err| FlagError::InvalidValue {
            flag: self.name.to_string(),
            value,
            position: self.position,
            reason: err.to_string(),
        })
    }

    /// Boolean flag: bare (`--jinja`) or explicit (`--jinja=false`).
    fn bool_flag(&self) -> Result<bool, FlagError> {
        match self.inline {
            None => Ok(true),
            Some(value) => match value.to_ascii_lowercase().as_str() {
                "1" | "true" | "on" | "yes" => Ok(true),
                "0" | "false" | "off" | "no" => Ok(false),
                _ => Err(FlagError::InvalidValue {
                    flag: self.name.to_string(),
                    value: value.to_string(),
                    position: self.position,
                    reason: "expected a boolean (true/false/on/off/1/0)".to_string(),
                }),
            },
        }
    }

    /// On/off flag that may be bare, inline, or take the next token.
    ///
    /// `-fa on` (production form), `-fa` (upstream form), `-fa=on` and
    /// `--fit off` are all accepted. A following token is only consumed when it
    /// does not start with `-`; `-fa --metrics` therefore means `-fa` turned on.
    fn switch(&self) -> Result<(Switch, usize), FlagError> {
        match self.inline {
            Some(value) => {
                let parsed =
                    Switch::parse_value(value).map_err(|reason| FlagError::InvalidValue {
                        flag: self.name.to_string(),
                        value: value.to_string(),
                        position: self.position,
                        reason,
                    })?;
                Ok((parsed, self.position + 1))
            }
            None => match self.argv.get(self.position + 1) {
                Some(next) if !next.starts_with('-') => {
                    let parsed =
                        Switch::parse_value(next).map_err(|reason| FlagError::InvalidValue {
                            flag: self.name.to_string(),
                            value: next.clone(),
                            position: self.position,
                            reason,
                        })?;
                    Ok((parsed, self.position + 2))
                }
                _ => Ok((Switch::On, self.position + 1)),
            },
        }
    }

    fn reasoning(&self) -> Result<(ReasoningMode, usize), FlagError> {
        match self.inline {
            Some(value) => {
                let parsed = ReasoningMode::parse_value(value).map_err(|reason| {
                    FlagError::InvalidValue {
                        flag: self.name.to_string(),
                        value: value.to_string(),
                        position: self.position,
                        reason,
                    }
                })?;
                Ok((parsed, self.position + 1))
            }
            None => match self.argv.get(self.position + 1) {
                Some(next) if !next.starts_with('-') => {
                    let parsed = ReasoningMode::parse_value(next).map_err(|reason| {
                        FlagError::InvalidValue {
                            flag: self.name.to_string(),
                            value: next.clone(),
                            position: self.position,
                            reason,
                        }
                    })?;
                    Ok((parsed, self.position + 2))
                }
                _ => Ok((ReasoningMode::Auto, self.position + 1)),
            },
        }
    }
}

/// Split `--port=8160` / `-c=200000` into name and inline value.
///
/// The `=` must not be the first character of the token, so a value token such as
/// `=x` is never mistaken for a flag.
fn split_flag(token: &str) -> (&str, Option<&str>) {
    match token.find('=') {
        Some(eq) if eq > 1 => (&token[..eq], Some(&token[eq + 1..])),
        _ => (token, None),
    }
}

/// A leading non-flag token is the program name when it looks like one.
fn looks_like_program(token: &str) -> bool {
    token.contains('/')
        || token == "llama-server"
        || token.ends_with("llama-server")
        || token.ends_with(".exe")
}

fn parse_f32_list(flag: &str, value: &str, position: usize) -> Result<Vec<f32>, FlagError> {
    let mut out = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        let parsed: f32 =
            trimmed
                .parse()
                .map_err(|err: std::num::ParseFloatError| FlagError::InvalidValue {
                    flag: flag.to_string(),
                    value: value.to_string(),
                    position,
                    reason: format!("`{trimmed}` is not a number: {err}"),
                })?;
        if !parsed.is_finite() {
            return Err(FlagError::InvalidValue {
                flag: flag.to_string(),
                value: value.to_string(),
                position,
                reason: format!("`{trimmed}` is not a finite number"),
            });
        }
        out.push(parsed);
    }
    if out.is_empty() {
        return Err(FlagError::InvalidValue {
            flag: flag.to_string(),
            value: value.to_string(),
            position,
            reason: "expected at least one number".to_string(),
        });
    }
    Ok(out)
}

/// First-seen bookkeeping for flags that may appear at most once.
#[derive(Default)]
struct Seen {
    entries: Vec<(&'static str, usize)>,
}

impl Seen {
    fn mark(&mut self, id: &'static str, flag: &str, position: usize) -> Result<(), FlagError> {
        if let Some((_, first_position)) = self.entries.iter().find(|(key, _)| *key == id) {
            return Err(FlagError::DuplicateFlag {
                flag: flag.to_string(),
                position,
                first_position: *first_position,
            });
        }
        self.entries.push((id, position));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::BackendHint;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_short_and_long_value_forms() {
        let plan =
            parse_llama_server_args(&argv(&["-c", "4096", "--port=8160", "-ngl=99"])).unwrap();
        assert_eq!(plan.ctx_size, Some(4096));
        assert_eq!(plan.port, Some(8160));
        assert_eq!(plan.gpu_layers, Some(99));
    }

    #[test]
    fn parses_long_aliases() {
        let plan = parse_llama_server_args(&argv(&[
            "--model",
            "m.gguf",
            "--n-gpu-layers",
            "99",
            "--ctx-size",
            "8192",
            "--parallel",
            "2",
            "--cache-type-k",
            "q8_0",
            "--threads",
            "8",
            "--ubatch-size",
            "128",
            "--flash-attn",
            "on",
            "--metrics",
            "--jinja",
            "--cpu-moe",
            "--kv-unified",
        ]))
        .unwrap();
        assert_eq!(plan.model, Some(PathBuf::from("m.gguf")));
        assert_eq!(plan.gpu_layers, Some(99));
        assert_eq!(plan.ctx_size, Some(8192));
        assert_eq!(plan.parallel, Some(2));
        assert_eq!(plan.cache_type_k, Some(KvCacheType::Q8_0));
        assert_eq!(plan.threads, Some(8));
        assert_eq!(plan.ubatch_size, Some(128));
        assert_eq!(plan.flash_attn, Some(Switch::On));
        assert!(plan.metrics && plan.jinja && plan.cpu_moe && plan.kv_unified);
    }

    #[test]
    fn negative_gpu_layers_is_a_value_not_a_flag() {
        let plan = parse_llama_server_args(&argv(&["-ngl", "-1"])).unwrap();
        assert_eq!(plan.gpu_layers, Some(-1));
    }

    #[test]
    fn bare_switch_flags_default_on() {
        let plan = parse_llama_server_args(&argv(&["-fa", "--metrics"])).unwrap();
        assert_eq!(plan.flash_attn, Some(Switch::On));
        assert!(plan.metrics);
    }

    #[test]
    fn host_expert_import_flag_parses_and_renders() {
        assert!(!parse_str("").unwrap().host_expert_import); // default OFF
        let plan = parse_str("--host-expert-import").unwrap();
        assert!(plan.host_expert_import);
        assert!(plan.render_args().contains(&"--host-expert-import".to_string()));
        let plan = parse_str("--host-expert-import=false").unwrap();
        assert!(!plan.host_expert_import);
        let err = parse_str("--host-expert-import --host-expert-import").unwrap_err();
        assert!(err.to_string().contains("duplicates"), "{err}");
    }

    #[test]
    fn bool_flags_accept_inline_false() {
        let plan = parse_llama_server_args(&argv(&["--jinja=false", "--metrics=0"])).unwrap();
        assert!(!plan.jinja);
        assert!(!plan.metrics);
        let plan = parse_llama_server_args(&argv(&["--jinja=true"])).unwrap();
        assert!(plan.jinja);
    }

    #[test]
    fn devices_parse_as_comma_list_and_append() {
        let plan =
            parse_llama_server_args(&argv(&["--device", "ROCm0,ROCm1", "--device=CPU"])).unwrap();
        assert_eq!(plan.devices.len(), 3);
        assert_eq!(plan.devices[0], DeviceRef::new(BackendHint::Hip, 0));
        assert_eq!(plan.devices[2], DeviceRef::cpu());
    }

    #[test]
    fn program_name_is_skipped_and_recorded() {
        let plan = parse_llama_server_args(&argv(&["/usr/bin/llama-server", "-c", "16"])).unwrap();
        assert_eq!(plan.binary, Some(PathBuf::from("/usr/bin/llama-server")));
        assert_eq!(plan.ctx_size, Some(16));

        // -D wins when both are present (program name is just a fallback).
        let plan =
            parse_llama_server_args(&argv(&["llama-server", "-D", "/other/server"])).unwrap();
        assert_eq!(plan.binary, Some(PathBuf::from("/other/server")));
    }

    #[test]
    fn unknown_flag_error_names_flag_and_position() {
        let err =
            parse_llama_server_args(&argv(&["-m", "x.gguf", "--frobnicate", "1"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unknown flag `--frobnicate` at argument position 2 (0-based)"
        );

        let err = parse_llama_server_args(&argv(&["-zzz"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unknown flag `-zzz` at argument position 0 (0-based)"
        );
    }

    #[test]
    fn missing_value_error_names_flag() {
        let err = parse_llama_server_args(&argv(&["-c"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "flag `-c` at argument position 0 expects a value"
        );
        let err = parse_llama_server_args(&argv(&["--host="])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "flag `--host` at argument position 0 expects a value"
        );
    }

    #[test]
    fn duplicate_flag_is_rejected_with_both_positions() {
        let err = parse_llama_server_args(&argv(&["-c", "1", "-c", "2"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "flag `-c` at argument position 2 duplicates the one at position 0"
        );
    }

    #[test]
    fn positional_argument_is_rejected() {
        let err = parse_llama_server_args(&argv(&["-c", "16", "stray"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unexpected positional argument `stray` at argument position 2 (0-based)"
        );
    }

    #[test]
    fn bad_values_name_flag_and_reason() {
        let err = parse_llama_server_args(&argv(&["--port", "70000"])).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("invalid value `70000` for flag `--port`"),
            "{text}"
        );
        assert!(text.contains("position 0"), "{text}");

        let err = parse_llama_server_args(&argv(&["-ctk", "q3_k"])).unwrap_err();
        assert!(err.to_string().contains("not a KV-cache type"), "{err}");

        let err = parse_llama_server_args(&argv(&["--backend", "metal"])).unwrap_err();
        assert!(err.to_string().contains("not a backend"), "{err}");

        let err = parse_llama_server_args(&argv(&["-ts", "9,,15"])).unwrap_err();
        assert!(err.to_string().contains("not a number"), "{err}");
    }

    #[test]
    fn bad_device_names_flag_position() {
        let err = parse_llama_server_args(&argv(&["--device", "ROCm-1"])).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("invalid device reference `ROCm-1`"), "{text}");
        assert!(text.contains("negative index"), "{text}");
        assert!(text.contains("position 0"), "{text}");
    }

    #[test]
    fn override_errors_are_indexed() {
        let err = parse_llama_server_args(&argv(&["-ot", r"a\.0=ROCm0,broken"])).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("entry #1"), "{text}");
        assert!(text.contains("expected `regex=device`"), "{text}");
    }

    #[test]
    fn tokenizer_handles_quotes_continuations_and_regex_escapes() {
        let tokens = tokenize("-m 'a b.gguf' --x \"c d\" blk\\.0 e\\\nf").unwrap();
        assert_eq!(
            tokens,
            vec![
                "-m".to_string(),
                "a b.gguf".to_string(),
                "--x".to_string(),
                "c d".to_string(),
                r"blk\.0".to_string(),
                "ef".to_string(),
            ]
        );
        let err = tokenize("-m 'unterminated").unwrap_err();
        assert!(err.to_string().contains("unterminated ' quote"), "{err}");
    }

    #[test]
    fn parse_str_matches_argv_parse() {
        let from_str = parse_str("-c 4096 --device ROCm0,ROCm1 -fa on").unwrap();
        let from_argv = parse_llama_server_args(&argv(&[
            "-c",
            "4096",
            "--device",
            "ROCm0,ROCm1",
            "-fa",
            "on",
        ]))
        .unwrap();
        assert_eq!(from_str, from_argv);
    }

    #[test]
    fn double_dash_terminator_only_allowed_at_end() {
        assert!(parse_llama_server_args(&argv(&["-c", "1", "--"])).is_ok());
        assert!(parse_llama_server_args(&argv(&["--", "-c"])).is_err());
    }
}
