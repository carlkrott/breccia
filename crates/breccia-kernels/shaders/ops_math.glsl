// Elementwise activation primitives shared by the Phase-3 op kernels.
//
// One definition each, so `silu(g)*u` (SwiGLU), the HC bottleneck activation and
// the gated-combine sigmoid cannot drift apart — the same reason
// `reference::dequant_block` exists for the decode units.
//
// Semantics are ggml's, transcribed:
//   * `GGML_SILU_FP32(x)  = x / (1 + expf(-x))`      (ggml/src/ggml.c, ggml_silu_f32)
//   * `GGML_SIGMOID_FP32(x) = 1 / (1 + expf(-x))`    (ggml/src/ggml.c, ggml_sigmoid_f32)
//
// Stability: `1/(1+exp(-x))` evaluates `exp(-x) = +inf` for `x` below about
// -88, and `x*0 = -0` there is only correct because `x/inf` is 0 — but `x*inf`
// style products (and a future `x*sigmoid` rewrite) would produce -inf*0 = NaN.
// The branch below keeps the two tails exact instead: sigmoid(-inf) = 0 and
// silu(-inf) = 0, which is also what the CPU reference returns for any input
// below -80 (where the true silu magnitude is < 1e-33, i.e. zero in f32).

#ifndef BRECCIA_OPS_MATH_GLSL
#define BRECCIA_OPS_MATH_GLSL

float breccia_sigmoid(float x) {
    if (x >= 0.0) {
        float e = exp(-x);
        return 1.0 / (1.0 + e);
    }
    float e = exp(x);
    return e / (1.0 + e);
}

float breccia_silu(float x) {
    // exp(80) ~ 5.5e34, so |silu(x)| < 1.4e-33 for x < -80: return the limit.
    if (x < -80.0) {
        return 0.0;
    }
    return x * breccia_sigmoid(x);
}

// tanh, with ggml's semantics (`tanhf`, ggml/src/ggml-cpu/vec.h ggml_vec_tanh_f32).
// GLSL `tanh` has no precision bound in Vulkan, so: odd Taylor polynomial for
// |x| < 0.25 (next term < 1e-8 relative), else (1 - e)/(1 + e) with
// e = exp(-2|x|) <= 0.61, which has no cancellation and no overflow.
float breccia_tanh(float x) {
    float a = abs(x);
    if (a < 0.25) {
        float x2 = x * x;
        return x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0 + x2 * (62.0 / 2835.0)))));
    }
    float e = exp(-2.0 * a);
    float t = (1.0 - e) / (1.0 + e);
    return x < 0.0 ? -t : t;
}

#endif // BRECCIA_OPS_MATH_GLSL
