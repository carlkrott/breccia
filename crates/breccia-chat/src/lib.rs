//! `breccia-chat` — model-agnostic chat formatting for `breccia serve`.
//!
//! * [`ChatTemplate`]: renders an OpenAI request (messages, `tools`, `tool_calls`,
//!   `tool` results, `reasoning_content`) through the model's own Jinja chat template
//!   (the GGUF `tokenizer.chat_template`, or an override file), the way Hugging Face
//!   `apply_chat_template` and llama.cpp `--jinja` do.
//! * [`parse`]: splits generated text into reasoning, content and OpenAI `tool_calls`,
//!   for every tool-call wire format we know (Qwen XML, Hermes JSON, DeepSeek
//!   V3/V3.1/V3.2, Kimi K2, GLM 4.5+, MiniMax M2, Seed-OSS, Mistral, Llama 3,
//!   Command-R, Granite, gpt-oss Harmony). The format is detected from the template.

pub mod parse;
pub mod template;

pub use parse::{parse, Format, Parsed, ToolCall};
pub use template::{ChatTemplate, Rendered};
