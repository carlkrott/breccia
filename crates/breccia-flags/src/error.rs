//! Errors produced while parsing a llama.cpp-compatible launch line.

use thiserror::Error;

/// A launch-line parse or validation failure.
///
/// Every variant carries enough context to point at the offending token: the
/// flag spelling as written, and the **0-based index** into the argv slice that
/// the flag occupied. `position` is therefore directly usable as `argv[position]`.
#[derive(Debug, Error)]
pub enum FlagError {
    /// A flag (or short alias) that `breccia-flags` does not model was found.
    ///
    /// Unknown flags are fatal on purpose: a silently dropped flag would change
    /// engine behaviour without the operator noticing.
    #[error("unknown flag `{flag}` at argument position {position} (0-based)")]
    UnknownFlag {
        /// Flag spelling as written, e.g. `--frobnicate`.
        flag: String,
        /// 0-based index of the flag in the argv slice.
        position: usize,
    },

    /// A bare token that is neither a flag nor the leading program name.
    #[error("unexpected positional argument `{value}` at argument position {position} (0-based)")]
    UnexpectedPositional {
        /// The token.
        value: String,
        /// 0-based index of the token in the argv slice.
        position: usize,
    },

    /// A flag that takes a value was the last token, or was written as `--x=`.
    #[error("flag `{flag}` at argument position {position} expects a value")]
    MissingValue {
        /// Flag spelling as written.
        flag: String,
        /// 0-based index of the flag in the argv slice.
        position: usize,
    },

    /// A value did not parse into the flag's type.
    #[error("invalid value `{value}` for flag `{flag}` at argument position {position}: {reason}")]
    InvalidValue {
        /// Flag spelling as written.
        flag: String,
        /// The rejected value.
        value: String,
        /// 0-based index of the flag in the argv slice.
        position: usize,
        /// Human-readable reason, e.g. `number too large to fit in target type`.
        reason: String,
    },

    /// A `--device` / `--mmdev` value was not a recognised device alias.
    #[error("invalid device reference `{value}` at argument position {position}: {reason}")]
    InvalidDevice {
        /// The rejected alias.
        value: String,
        /// 0-based index of the flag in the argv slice.
        position: usize,
        /// Human-readable reason.
        reason: String,
    },

    /// The same flag appeared twice where only one occurrence is meaningful.
    #[error(
        "flag `{flag}` at argument position {position} duplicates the one at position {first_position}"
    )]
    DuplicateFlag {
        /// Flag spelling as written (second occurrence).
        flag: String,
        /// 0-based index of the second occurrence.
        position: usize,
        /// 0-based index of the first occurrence.
        first_position: usize,
    },

    /// A single `-ot` entry was structurally invalid (missing `=`, empty side).
    #[error("-ot override entry #{index} `{entry}` is invalid: {reason}")]
    OverrideEntry {
        /// 0-based index into the *accumulated* override list.
        index: usize,
        /// The offending entry, without surrounding quotes.
        entry: String,
        /// Human-readable reason.
        reason: String,
    },

    /// An `-ot` override pattern failed to compile as a regular expression.
    #[error("-ot override entry #{index} `{entry}` has an invalid regex: {reason}")]
    OverrideRegex {
        /// 0-based index into the *accumulated* override list.
        index: usize,
        /// The offending entry.
        entry: String,
        /// `regex` crate error text.
        reason: String,
    },

    /// The `parse_str` helper could not split the input into tokens.
    #[error("could not tokenize launch line at token {position}: {reason}")]
    Tokenize {
        /// Number of complete tokens read before the failure.
        position: usize,
        /// Human-readable reason, e.g. `unterminated ' quote`.
        reason: String,
    },
}
