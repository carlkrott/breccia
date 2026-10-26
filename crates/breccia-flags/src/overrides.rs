//! `-ot` (per-tensor device override) parsing.
//!
//! The production override is a single argv element holding a comma-separated
//! list of `regex=device` entries:
//!
//! ```text
//! blk\.(0|1|2|4|8|18|20|22|23)\.ffn_(up|down|gate)_exps\.=ROCm0,blk\.(24|35)\.ffn_(up|down|gate)_exps\.=ROCm1
//! ```
//!
//! Splitting on raw commas is wrong in general: a regex may contain commas in a
//! group (`blk\.(1,2)\.`), a character class (`[a,b]`) or a counted repetition
//! (`\d{1,3}`). The splitter here only breaks on a separator at nesting depth 0
//! and never inside an escape, and the `regex=device` split uses the **last**
//! top-level `=` so a pattern containing `=` still parses.

use regex::Regex;

use crate::device::DeviceRef;
use crate::error::FlagError;

/// Split `input` on `separator`, ignoring separators inside `()`, `[]`, `{}` or
/// escaped with a backslash.
///
/// Nesting is tracked leniently: an unmatched closing bracket cannot drive the
/// depth below zero, so malformed patterns degrade to "fewer splits", never to a
/// panic.
pub fn split_top_level(input: &str, separator: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth: i32 = 0;
    let mut escaped = false;

    for ch in input.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => {
                current.push(ch);
                escaped = true;
            }
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    depth = 0;
                }
                current.push(ch);
            }
            c if c == separator && depth == 0 => parts.push(std::mem::take(&mut current)),
            c => current.push(c),
        }
    }
    parts.push(current);
    parts
}

/// Split one `regex=device` entry at its last top-level `=`.
///
/// Returns trimmed `(pattern, device_alias)` or a human-readable reason.
pub fn split_override_entry(entry: &str) -> Result<(String, String), String> {
    let mut depth: i32 = 0;
    let mut escaped = false;
    let mut split_at: Option<usize> = None;

    for (idx, ch) in entry.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    depth = 0;
                }
            }
            '=' if depth == 0 => split_at = Some(idx),
            _ => {}
        }
    }

    let at = split_at.ok_or_else(|| "expected `regex=device`".to_string())?;
    let pattern = entry[..at].trim();
    let device = entry[at + 1..].trim();
    if pattern.is_empty() {
        return Err("empty regex before `=`".to_string());
    }
    if device.is_empty() {
        return Err("empty device after `=`".to_string());
    }
    Ok((pattern.to_string(), device.to_string()))
}

/// Parse an `-ot` value into structured entries.
///
/// `base_index` is the number of overrides already collected by earlier `-ot`
/// occurrences, so error indices point into the accumulated list.
pub fn parse_override_list(
    value: &str,
    base_index: usize,
) -> Result<Vec<(String, DeviceRef)>, FlagError> {
    let mut out = Vec::new();
    for (offset, entry) in split_top_level(value, ',').iter().enumerate() {
        let index = base_index + offset;
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            return Err(FlagError::OverrideEntry {
                index,
                entry: entry.clone(),
                reason: "empty override entry (stray comma?)".to_string(),
            });
        }
        let (pattern, device) =
            split_override_entry(trimmed).map_err(|reason| FlagError::OverrideEntry {
                index,
                entry: trimmed.to_string(),
                reason,
            })?;
        let device = DeviceRef::parse(&device).map_err(|reason| FlagError::OverrideEntry {
            index,
            entry: trimmed.to_string(),
            reason,
        })?;
        out.push((pattern, device));
    }
    Ok(out)
}

/// An `-ot` entry whose pattern is compiled and ready to match tensor names.
#[derive(Debug)]
pub struct CompiledOverride {
    /// Pattern source, exactly as written on the launch line.
    pub pattern: String,
    /// Compiled regular expression.
    pub regex: Regex,
    /// Device that tensors matching `regex` are placed on.
    pub device: DeviceRef,
}

impl CompiledOverride {
    /// `true` when `tensor_name` matches this override (unanchored, like llama.cpp).
    pub fn matches(&self, tensor_name: &str) -> bool {
        self.regex.is_match(tensor_name)
    }
}

/// Compile every `-ot` pattern, reporting **all** failures with their indices.
///
/// This is the validation step the loader should run once at startup: it turns
/// the parse-time string list into matchers, and a bad regex is reported by
/// position in the accumulated override list rather than as a bare `regex` error.
pub fn validate_overrides(
    entries: &[(String, DeviceRef)],
) -> Result<Vec<CompiledOverride>, Vec<FlagError>> {
    let mut compiled = Vec::with_capacity(entries.len());
    let mut errors = Vec::new();

    for (index, (pattern, device)) in entries.iter().enumerate() {
        match Regex::new(pattern) {
            Ok(regex) => compiled.push(CompiledOverride {
                pattern: pattern.clone(),
                regex,
                device: *device,
            }),
            Err(err) => errors.push(FlagError::OverrideRegex {
                index,
                entry: format!("{pattern}={device}"),
                reason: err.to_string(),
            }),
        }
    }

    if errors.is_empty() {
        Ok(compiled)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::BackendHint;

    #[test]
    fn splits_plain_list() {
        assert_eq!(
            split_top_level("a=1,b=2,c=3", ','),
            vec!["a=1".to_string(), "b=2".to_string(), "c=3".to_string()]
        );
        assert_eq!(split_top_level("", ','), vec![String::new()]);
    }

    #[test]
    fn does_not_split_inside_parens() {
        let s = r"blk\.(0|1|2)\.x=ROCm0,blk\.(24|35)\.y=ROCm1";
        let parts = split_top_level(s, ',');
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], r"blk\.(0|1|2)\.x=ROCm0");
        assert_eq!(parts[1], r"blk\.(24|35)\.y=ROCm1");
    }

    #[test]
    fn does_not_split_inside_pipes_groups_or_repetition() {
        // comma inside a group alternation
        assert_eq!(
            split_top_level(r"blk\.(1,2)\.w=ROCm0", ','),
            vec![r"blk\.(1,2)\.w=ROCm0".to_string()]
        );
        // comma inside a character class
        assert_eq!(
            split_top_level(r"[a,b]+\.w=ROCm0", ','),
            vec![r"[a,b]+\.w=ROCm0".to_string()]
        );
        // comma inside a counted repetition
        assert_eq!(
            split_top_level(r"x\d{1,3}\.w=ROCm0,y=ROCm1", ','),
            vec![r"x\d{1,3}\.w=ROCm0".to_string(), "y=ROCm1".to_string()]
        );
        // comma inside nested groups
        assert_eq!(
            split_top_level(r"a\.(b|(c,(d)))\.w=ROCm0,z=CPU", ','),
            vec![r"a\.(b|(c,(d)))\.w=ROCm0".to_string(), "z=CPU".to_string()]
        );
        // escaped comma is literal
        assert_eq!(
            split_top_level(r"a\,b=ROCm0", ','),
            vec![r"a\,b=ROCm0".to_string()]
        );
        // pipes at top level are not separators
        assert_eq!(
            split_top_level(r"a|b=ROCm0", ','),
            vec![r"a|b=ROCm0".to_string()]
        );
    }

    #[test]
    fn unmatched_closing_bracket_cannot_underflow() {
        let parts = split_top_level(r"a)b,c=d", ',');
        assert_eq!(parts, vec!["a)b".to_string(), "c=d".to_string()]);
    }

    #[test]
    fn splits_entry_at_last_top_level_equals() {
        assert_eq!(
            split_override_entry("blk\\.0\\.ffn_up_exps\\.=ROCm0").unwrap(),
            (r"blk\.0\.ffn_up_exps\.".to_string(), "ROCm0".to_string())
        );
        // a pattern containing '=' keeps the last top-level '=' as separator
        assert_eq!(
            split_override_entry(r"a=b=ROCm0").unwrap(),
            ("a=b".to_string(), "ROCm0".to_string())
        );
        // '=' inside a group belongs to the pattern
        assert_eq!(
            split_override_entry(r"a(=|=)b=CPU").unwrap(),
            (r"a(=|=)b".to_string(), "CPU".to_string())
        );
    }

    #[test]
    fn rejects_bad_entries() {
        assert!(split_override_entry("no_equals").is_err());
        assert!(split_override_entry("=ROCm0").is_err());
        assert!(split_override_entry("a=").is_err());
        assert!(parse_override_list("", 0).is_err());
        assert!(parse_override_list("a=ROCm0,,b=ROCm1", 0).is_err());
    }

    #[test]
    fn indices_are_accumulated_and_devices_typed() {
        let entries = parse_override_list(r"a\.0=ROCm0,b\.1=Vulkan1", 4).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, r"a\.0");
        assert_eq!(entries[0].1.backend_hint, BackendHint::Hip);
        assert_eq!(entries[1].1.backend_hint, BackendHint::Vulkan);

        let err = parse_override_list("a=ROCm-1", 7).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("entry #7"), "{text}");
    }

    #[test]
    fn validation_reports_all_bad_regexes_with_indices() {
        let entries = vec![
            (r"blk\.0\.ok\.".to_string(), DeviceRef::cpu()),
            ("(unclosed".to_string(), DeviceRef::cpu()),
            ("[z-a]".to_string(), DeviceRef::cpu()),
        ];
        let errors = validate_overrides(&entries).unwrap_err();
        assert_eq!(errors.len(), 2);
        assert!(
            errors[0].to_string().contains("entry #1"),
            "{:?}",
            errors[0]
        );
        assert!(
            errors[1].to_string().contains("entry #2"),
            "{:?}",
            errors[1]
        );
    }

    #[test]
    fn compiled_override_matches_tensor_names() {
        let entries = vec![(
            r"blk\.(0|1|2)\.ffn_(up|down|gate)_exps\.".to_string(),
            DeviceRef::parse("ROCm1").unwrap(),
        )];
        let compiled = validate_overrides(&entries).unwrap();
        assert!(compiled[0].matches("blk.2.ffn_gate_exps.weight"));
        assert!(!compiled[0].matches("blk.3.ffn_gate_exps.weight"));
    }
}
