//! End-to-end tests against the **verbatim** production launch line
//! (`qwen-dual-serve.sh`), plus round-trip and error-surface checks.
//!
//! The point of this file: the exact string the production service uses must
//! parse into a plan, survive render→parse unchanged, and resolve `ROCm0/ROCm1`
//! to Vulkan devices while staying launchable.

use breccia_flags::{
    parse_llama_server_args, parse_str, tokenize, Backend, BackendHint, DeviceRef, KvCacheType,
    LaunchPlan, ReasoningMode, Switch, WarningCode,
};
use std::path::PathBuf;
use std::sync::OnceLock;

static CELL: OnceLock<String> = OnceLock::new();

/// Verbatim argument list from the production `qwen-dual-serve.sh`
/// (paths supplied via BRECCIA_MODEL_ROOT / BRECCIA_SLOT_ROOT).
///
/// Only the shell's line continuations are omitted; every flag and value is
/// character-for-character what the service passes to `llama-server`.
const SERVICE_ARGS_TEMPLATE: &str = r#"
-D <MODEL_ROOT>/bin/llama-server
-m /srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
--device ROCm0,ROCm1 -ngl 99 -ts 9,15
-ot 'blk\.(0|1|2|4|8|18|20|22|23)\.ffn_(up|down|gate)_exps\.=ROCm0,blk\.(24|35)\.ffn_(up|down|gate)_exps\.=ROCm1'
-cmoe --moe-cache-mib 0 -lzm on -lm none --no-host
-c 200000 -kvu -np 1 --kv-unified-per-slot 200000 -ctk q4_0 -ctv q4_0 -fa on
-cram 0 -ctxcp 1 --slot-save-path <SLOT_ROOT>/kv-slots/IQ3_XXS-q4_0-rocm-1x200k-1b70350c9/
--mmproj /srv/strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf -mmdev ROCm0 --image-min-tokens 1024 --image-max-tokens 2048
-lv 4 --reasoning off
-t 16 -tb 16 --host 127.0.0.1 --port 8160 --jinja --metrics --fit off --no-op-offload -ub 256
"#;

fn path_var(name: &'static str) -> &'static str {
    CELL.get_or_init(|| {
        match std::env::var(name) {
            Ok(v) if !v.trim().is_empty() => v.trim_end_matches('/').to_string(),
            _ => panic!("{name} must be set for launch_surface tests"),
        }
    }).as_str()
}

fn service_args() -> String {
    SERVICE_ARGS_TEMPLATE
        .replace("<MODEL_ROOT>", path_var("BRECCIA_MODEL_ROOT"))
        .replace("<SLOT_ROOT>", path_var("BRECCIA_SLOT_ROOT"))
}

fn service_plan() -> LaunchPlan {
    parse_str(&service_args()).expect("production launch line must parse")
}

#[test]
fn parses_the_verbatim_production_line() {
    let plan = service_plan();

    // Backend defaults to Vulkan even though every device alias is HIP-flavoured.
    assert_eq!(plan.backend, Backend::Vulkan);

    assert_eq!(
        plan.binary,
        Some(PathBuf::from(
            format!("{}/bin/llama-server", path_var("BRECCIA_MODEL_ROOT"))
        ))
    );
    assert_eq!(
        plan.model,
        Some(PathBuf::from(
            "/srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
        ))
    );
    assert_eq!(
        plan.devices,
        vec![
            DeviceRef::new(BackendHint::Hip, 0),
            DeviceRef::new(BackendHint::Hip, 1)
        ]
    );
    assert_eq!(plan.gpu_layers, Some(99));
    assert_eq!(plan.tensor_split, Some(vec![9.0, 15.0]));

    // -ot: two entries, both patterns intact including escapes and pipes.
    assert_eq!(plan.override_tensors.len(), 2);
    assert_eq!(
        plan.override_tensors[0].0,
        r"blk\.(0|1|2|4|8|18|20|22|23)\.ffn_(up|down|gate)_exps\."
    );
    assert_eq!(
        plan.override_tensors[0].1,
        DeviceRef::new(BackendHint::Hip, 0)
    );
    assert_eq!(
        plan.override_tensors[1].0,
        r"blk\.(24|35)\.ffn_(up|down|gate)_exps\."
    );
    assert_eq!(
        plan.override_tensors[1].1,
        DeviceRef::new(BackendHint::Hip, 1)
    );

    // Fork-specific surface.
    assert!(plan.cpu_moe);
    assert_eq!(plan.moe_cache_mib, Some(0));
    assert_eq!(plan.lzm, Some(Switch::On));
    assert_eq!(plan.lm.as_deref(), Some("none"));
    assert!(plan.no_host);
    assert!(plan.kv_unified);
    assert_eq!(plan.kv_unified_per_slot, Some(200_000));
    assert_eq!(plan.cram, Some(0));
    assert_eq!(plan.ctxcp, Some(1));
    assert_eq!(plan.fit, Some(Switch::Off));
    assert!(plan.no_op_offload);
    assert_eq!(plan.log_verbosity, Some(4));
    assert_eq!(plan.reasoning, Some(ReasoningMode::Off));

    // Standard surface.
    assert_eq!(plan.ctx_size, Some(200_000));
    assert_eq!(plan.parallel, Some(1));
    assert_eq!(plan.cache_type_k, Some(KvCacheType::Q4_0));
    assert_eq!(plan.cache_type_v, Some(KvCacheType::Q4_0));
    assert_eq!(plan.flash_attn, Some(Switch::On));
    assert_eq!(
        plan.slot_save_path,
        Some(PathBuf::from(
            format!("{}/kv-slots/IQ3_XXS-q4_0-rocm-1x200k-1b70350c9/", path_var("BRECCIA_SLOT_ROOT"))
        ))
    );
    assert_eq!(
        plan.mmproj,
        Some(PathBuf::from(
            "/srv/strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
        ))
    );
    assert_eq!(
        plan.mmproj_device,
        Some(DeviceRef::new(BackendHint::Hip, 0))
    );
    assert_eq!(plan.image_min_tokens, Some(1024));
    assert_eq!(plan.image_max_tokens, Some(2048));
    assert_eq!(plan.threads, Some(16));
    assert_eq!(plan.threads_batch, Some(16));
    assert_eq!(plan.host.as_deref(), Some("127.0.0.1"));
    assert_eq!(plan.port, Some(8160));
    assert!(plan.jinja);
    assert!(plan.metrics);
    assert_eq!(plan.ubatch_size, Some(256));
}

#[test]
fn every_override_pattern_compiles() {
    let plan = service_plan();
    let compiled = plan
        .validated_overrides()
        .expect("production -ot patterns must compile");
    assert_eq!(compiled.len(), 2);
    // Placement actually works the way the service intends.
    assert!(compiled[0].matches("blk.22.ffn_down_exps.weight"));
    assert!(!compiled[0].matches("blk.24.ffn_down_exps.weight"));
    assert!(compiled[1].matches("blk.35.ffn_up_exps.weight"));
}

#[test]
fn rocm_aliases_resolve_to_vulkan_with_warnings() {
    let plan = service_plan();
    let resolution = plan.resolve_devices_detailed();

    assert_eq!(resolution.resolved.len(), 2);
    for (index, device) in resolution.resolved.iter().enumerate() {
        assert_eq!(device.backend, BackendHint::Vulkan, "device {index}");
        assert_eq!(device.index, index as u32);
        assert!(device.is_remapped());
    }
    assert_eq!(
        plan.resolve_devices(),
        resolution.resolved,
        "the literal-signature API must agree"
    );

    assert_eq!(resolution.warnings.len(), 2);
    for warning in &resolution.warnings {
        assert_eq!(warning.code, WarningCode::HipAliasOnVulkan);
        assert!(warning.message.contains("backend stays vulkan"));
    }

    // The mmproj device follows the same rule.
    let mmdev = plan.mmproj_device.unwrap();
    assert_eq!(mmdev.backend_hint, BackendHint::Hip);
}

#[test]
fn switching_to_hip_keeps_the_same_devices() {
    let mut plan = service_plan();
    plan.backend = Backend::Hip;
    let resolution = plan.resolve_devices_detailed();
    assert!(resolution.warnings.is_empty());
    assert_eq!(resolution.resolved[0].backend, BackendHint::Hip);
    assert_eq!(resolution.resolved[1].backend, BackendHint::Hip);
    assert_eq!(resolution.resolved[1].index, 1);
}

#[test]
fn round_trip_parse_render_parse_is_a_fixed_point() {
    let plan = service_plan();
    let rendered = plan.render_args();
    let reparsed = parse_llama_server_args(&rendered).expect("rendered argv must re-parse");
    assert_eq!(plan, reparsed, "rendered argv must describe the same plan");
    assert_eq!(
        rendered,
        reparsed.render_args(),
        "rendering must be canonical/idempotent"
    );

    // ... and through the string form too.
    let as_line = plan.render_args().join(" ");
    assert_eq!(parse_str(&as_line).unwrap(), plan);
}

#[test]
fn argv_and_string_forms_agree() {
    let from_str = service_plan();
    let argv = tokenize(&service_args()).unwrap();
    let from_argv = parse_llama_server_args(&argv).unwrap();
    assert_eq!(from_str, from_argv);
    assert_eq!(argv[0], "-D");
    assert_eq!(
        argv.len(),
        66,
        "unexpected token count for the service line"
    );
}

#[test]
fn inline_equals_form_is_equivalent() {
    // Rewrite every `flag value` pair from the canonical render into `--flag=value`
    // and check we land on the identical plan. Boolean flags stay bare.
    let plan = service_plan();
    let rendered = plan.render_args();
    let mut inline: Vec<String> = Vec::new();
    let mut i = 0;
    while i < rendered.len() {
        let token = &rendered[i];
        let has_pair = rendered
            .get(i + 1)
            .map(|v| !v.starts_with('-'))
            .unwrap_or(false);
        if has_pair {
            inline.push(format!("{token}={}", rendered[i + 1]));
            i += 2;
        } else {
            inline.push(token.clone());
            i += 1;
        }
    }
    let from_inline = parse_llama_server_args(&inline).expect("inline form must parse");
    assert_eq!(from_inline, plan);
}

#[test]
fn unknown_flag_fails_with_flag_and_position() {
    let args = vec![
        "-m".to_string(),
        "model.gguf".to_string(),
        "--experimental-thing".to_string(),
        "7".to_string(),
    ];
    let err = parse_llama_server_args(&args).unwrap_err();
    assert_eq!(
        err.to_string(),
        "unknown flag `--experimental-thing` at argument position 2 (0-based)"
    );

    // A single unknown flag in the middle of the production line still fails
    // loudly, and the position is the real argv index.
    let mut argv = tokenize(&service_args()).unwrap();
    let insert_at = argv.iter().position(|t| t == "--jinja").unwrap();
    argv.insert(insert_at, "--no-such-flag".to_string());
    let err = parse_llama_server_args(&argv).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("unknown flag `--no-such-flag`"), "{text}");
    assert!(
        text.contains(&format!("argument position {insert_at}")),
        "{text}"
    );
}

#[test]
fn plan_json_round_trips_with_overrides_and_warnings() {
    let plan = service_plan();
    let json = serde_json::to_string_pretty(&plan).unwrap();
    let back: LaunchPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(plan, back);

    let warnings = serde_json::to_string(&plan.warnings()).unwrap();
    assert!(warnings.contains("hip_alias_on_vulkan"), "{warnings}");

    // A plan serialised without one field falls back to the llama.cpp default.
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("moe_cache_mib");
    let partial: LaunchPlan = serde_json::from_value(value).unwrap();
    assert_eq!(partial.moe_cache_mib, None);
    assert_eq!(partial.override_tensors, plan.override_tensors);
}

#[test]
fn malformed_lines_never_panic() {
    // Every prefix of the production line must either parse or return an error.
    let argv = tokenize(&service_args()).unwrap();
    for cut in 0..=argv.len() {
        let _ = parse_llama_server_args(&argv[..cut]);
    }
    for bad in [
        "-ot",
        "-ot =ROCm0",
        "-ot a=",
        "--device ,",
        "--device ROCm0,",
        "-ts ,",
        "-c notanumber",
        "--port -1",
        "-ctk",
        "--",
        "-",
        "",
        "   ",
        "'unterminated",
        "\"unterminated",
    ] {
        let _ = parse_str(bad);
    }
}
