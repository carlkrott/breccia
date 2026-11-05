#!/bin/bash
# Stage-2 vision: breccia serve (default devices) + CPU encoder sidecar; POST 448/1024/2048 images as data URLs.
set -u; cd /var/tmp/breccia-coord; export GATE_SINCE="2026-11-04 21:12"; U=r367-coord-vserve; L=$PWD/out/unit-vserve.log; V=$PWD/vision
rm -f $L
if ! GATE=$(bash /var/tmp/breccia-pc2/gate.sh); then echo "GATE FAIL $GATE"; exit 90; fi
systemd-inhibit --what=sleep:idle --who=breccia --why="vision serve test" --mode=block \
  systemd-run --user --unit=$U --collect -p MemoryMax=48G -p MemorySwapMax=0 -p LimitMEMLOCK=infinity \
  -p StandardOutput=append:$L -p StandardError=append:$L \
  $PWD/out/breccia-vis serve --port 8164 --ctx 4096 --max-tokens 256 \
  --mmproj /srv/strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf
(bash hangwatch.sh $U $L 600) & HW=$!
for i in $(seq 120); do curl -sf -m 2 localhost:8164/health >/dev/null && break; systemctl --user is-active -q $U || break; sleep 3; done
curl -sf -m 2 localhost:8164/health >/dev/null || { echo "SERVE NOT UP"; tail -n 20 $L; systemctl --user stop $U; kill $HW; echo VSERVE DONE; exit 1; }
python3 - "$V" <<'PY'
import base64, json, sys, time, urllib.request
V = sys.argv[1]
def ask(content, n=160):
    req = urllib.request.Request("http://127.0.0.1:8164/v1/chat/completions", method="POST",
        data=json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": n}).encode(),
        headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        return time.time() - t, e.code, e.read().decode()[:200]
    return time.time() - t, r["usage"], r["choices"][0]["message"]["content"]
for name in ("img448", "img1024", "img2048"):
    url = "data:image/png;base64," + base64.b64encode(open(f"{V}/{name}.png", "rb").read()).decode()
    dt, u, txt = ask([{"type": "image_url", "image_url": {"url": url}},
                      {"type": "text", "text": "Describe this image in one sentence."}])
    print(f"[{name}] {dt:.1f}s usage={u}\n  {txt!r}", flush=True)
dt, u, txt = ask("Say hello in five words.", 32)
print(f"[text-after] {dt:.1f}s usage={u}\n  {txt!r}", flush=True)
PY
systemctl --user stop $U; kill $HW 2>/dev/null
grep -E "image  |vision |panic|error|HANGWATCH|pin failed" $L | head -20
echo VSERVE DONE
