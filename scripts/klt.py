#!/usr/bin/env python3
"""Teacher-forced tail gate: mean/max KL(ref||x), mean TV, top-1 agreement over K rows of .tail dumps."""
import sys, numpy as np
V = 248320
def rows(p):
    v = np.fromfile(p, dtype=np.float32).astype(np.float64).reshape(-1, V)
    v -= v.max(1, keepdims=True); return v - np.log(np.exp(v).sum(1, keepdims=True))
a, b = rows(sys.argv[1]), rows(sys.argv[2])
pa, pb = np.exp(a), np.exp(b)
kl = (pa * (a - b)).sum(1); tv = 0.5 * np.abs(pa - pb).sum(1)
t1 = (a.argmax(1) == b.argmax(1)).mean()
t10 = np.mean([len(set(np.argsort(-x)[:10]) & set(np.argsort(-y)[:10])) / 10 for x, y in zip(a, b)])
print(f"{sys.argv[2]}: K={len(a)} KL mean {kl.mean():.3e} max {kl.max():.3e} | TV mean {tv.mean():.4f} max {tv.max():.4f} | top1 {t1:.3f} top10 {t10:.3f}")
