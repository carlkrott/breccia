#!/usr/bin/env python3
"""Cross-stream expert overlap from single-stream decode route traces (d rows).
Usage: overlap.py <gpu-blocks RX> <gpu-blocks MI50> route-s0.txt ... (stream order; stream 1 = MI50)"""
import sys, collections
rx_gpu = {int(x) for x in sys.argv[1].split(',')}; mi_gpu = {int(x) for x in sys.argv[2].split(',')}
files = sys.argv[3:]
traces = []
for f in files:
    t = {}
    for line in open(f):
        if not line.startswith('d'): continue
        p = line.split(); pos = int(p[0][1:]); blk = int(p[1])
        t[(pos, blk)] = {int(e.split(':')[0]) for e in p[2:]}
    traces.append(t)
keys = set.intersection(*(set(t) for t in traces))
def stat(streams, gpu_of):
    tot = dist = 0; per_blk = collections.Counter()
    for k in keys:
        sets = [traces[s][k] for s in streams if k[1] not in gpu_of(s)]
        if len(sets) < 2: continue
        tot += sum(len(x) for x in sets); dist += len(set().union(*sets))
    return tot, dist
gpu_of = lambda s: mi_gpu if s == 1 else rx_gpu
for name, ss in [('RX lockstep {0,2,3}', [0, 2, 3]), ('all 4 shared pool', [0, 1, 2, 3])]:
    tot, dist = stat(ss, gpu_of)
    print(f"{name}: host expert-loads {tot}, distinct {dist}, saving {1 - dist/tot:.1%}" if tot else f"{name}: none")
k = len(next(iter(traces[0].values()))); print(f"keys {len(keys)} (pos x block), top-k {k}, blocks {len({b for _, b in keys})}")
