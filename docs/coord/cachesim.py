#!/usr/bin/env python3
"""Expert-cache replay over route traces (d<pos> <block> e:w x10).
Compares a dynamic VRAM expert cache with the static whole-block arena at the
same byte budget. Usage: cachesim.py <budget_MiB> <route files...>"""
import sys
from collections import OrderedDict

MIB_PER_EXPERT = 40925 / (48 * 512)  # host arena average; per-block quant types vary (blk.0 IQ2_XS 1.78)

def load(path):
    toks = {}
    for line in open(path):
        p = line.split()
        toks.setdefault(int(p[0][1:]), []).append((int(p[1]), [int(x.split(':')[0]) for x in p[2:]]))
    return [toks[k] for k in sorted(toks)]

def lru(streams, cap, warm, per_block=None):
    """Interleave streams token by token into ONE cache (one arena per card).
    cap = total experts; per_block splits it evenly per block when set."""
    caches = {}
    hit = miss = 0
    n = min(len(s) for s in streams)
    for t in range(n):
        for s in streams:
            for b, es in s[t]:
                key = b if per_block else 0
                c = caches.setdefault(key, OrderedDict())
                lim = per_block if per_block else cap
                for e in es:
                    k = (b, e)
                    if k in c:
                        c.move_to_end(k)
                        if t >= warm: hit += 1
                    else:
                        if t >= warm: miss += 1
                        c[k] = 1
                        if len(c) > lim: c.popitem(last=False)
    return hit / (hit + miss)

def topn_upper(streams, cap):
    """In-sample best static expert set of size cap (upper bound: it saw the future)."""
    from collections import Counter
    cnt = Counter((b, e) for s in streams for tok in s for b, es in tok for e in es)
    tot = sum(cnt.values())
    return sum(v for _, v in cnt.most_common(cap)) / tot

if __name__ == '__main__':
    budget = float(sys.argv[1])
    streams = [load(p) for p in sys.argv[2:]]
    # self-check: a cache that never evicts misses only first touches
    refs = [(b, e) for s in streams for tok in s[:min(len(x) for x in streams)] for b, es in tok for e in es]
    assert abs(lru(streams, len(refs), 0) - (1 - len(set(refs)) / len(refs))) < 1e-9
    cap = int(budget / MIB_PER_EXPERT)
    blocks = int(budget // (512 * MIB_PER_EXPERT))
    print(f'budget {budget:.0f} MiB = {cap} experts = {blocks} whole blocks; streams {len(streams)}')
    print(f'  static whole-block arena  hit {blocks/48:6.1%}')
    print(f'  LRU per-block cold        hit {lru(streams, cap, 0, cap // 48):6.1%}')
    print(f'  LRU per-block warm(t>=80) hit {lru(streams, cap, 80, cap // 48):6.1%}')
    print(f'  LRU global warm(t>=80)    hit {lru(streams, cap, 80):6.1%}')
    print(f'  top-N in-sample (upper)   hit {topn_upper(streams, cap):6.1%}')
