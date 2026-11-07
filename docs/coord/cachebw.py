#!/usr/bin/env python3
"""Per-expert VRAM cache with a capped promotion budget (decayed counts, swap coldest for hottest
every N tokens, RS expert_cache.rs style). Reports warm hit rate, upload MiB/token, and T-row distinct
misses. Usage: cachebw.py <budget_MiB> <files...>"""
import sys, heapq
from collections import defaultdict
sys.path.insert(0, sys.argv[0].rsplit('/',1)[0])
MIB = 40925/(48*512)
def load(path):
    toks = {}
    for line in open(path):
        p = line.split()
        toks.setdefault(int(p[0][1:]), []).append((int(p[1]), [int(x.split(':')[0]) for x in p[2:]]))
    return [toks[k] for k in sorted(toks)]
def run(streams, cap, every, per_adapt, keep=0.95, warm=80, T=1):
    res=set(); cnt=defaultdict(float); up=0; hit=miss=0; n=min(map(len,streams)); dm=dt=0
    for t in range(n):
        if T>1 and t%T==0: seen=[set() for _ in streams]
        for si,s in enumerate(streams):
            for b,es in s[t]:
                for e in es:
                    k=(b,e); cnt[k]+=1
                    h=k in res
                    if t>=warm:
                        hit+=h; miss+=(not h)
                        if T>1:
                            if k not in seen[si]:
                                seen[si].add(k); dt+=1; dm+=(not h)
        if (t+1)%every==0:
            free=cap-len(res)
            cand=heapq.nlargest(per_adapt,(k for k in cnt if k not in res),key=cnt.get)
            cold=heapq.nsmallest(per_adapt,res,key=lambda k:cnt.get(k,0)) if len(cand)>free else []
            ci=0
            for k in cand:
                if free>0: res.add(k); free-=1; up+=(t>=warm)
                elif ci<len(cold) and cnt[k]>cnt.get(cold[ci],0)*1.5:
                    res.discard(cold[ci]); ci+=1; res.add(k); up+=(t>=warm)
            for k in list(cnt):
                cnt[k]*=keep
                if cnt[k]<1e-3: del cnt[k]
    w=max(1,n-warm)*len(streams)
    return hit/(hit+miss), up*MIB/w, (dm/dt if dt else 0), hit+miss, w
budget=float(sys.argv[1]); cap=int(budget/MIB)
streams=[load(p) for p in sys.argv[2:]]
print(f'cap {cap} experts, tokens {min(map(len,streams))} streams {len(streams)} refs/token {sum(len(es) for t in streams[0][:50] for _,es in t)/50:.0f}')
print('every per_adapt   hit   upMiB/tok  distinct-miss(T=4)')
for every,pa in [(8,8),(8,32),(8,128),(8,512),(4,256),(1,16),(1,64),(1,4096)]:
    h,u,dm,_,_=run(streams,cap,every,pa,T=4)
    print(f'{every:5d} {pa:6d} {h:6.1%} {u:9.1f}   {dm:6.1%}')
