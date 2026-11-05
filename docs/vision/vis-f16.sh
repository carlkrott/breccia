#!/bin/bash
# q6 (default) vs f16 KV on image prompts; q6 arms = out/report-vi-448.json / vi-1024.json
set -u; cd /var/tmp/breccia-coord; source <(sed -n '/^set -u/,/^}/p' vis-job.sh)
arm vi-f16 out/breccia-vis $V/prompt-img.ids 48 1024 --setenv=BRECCIA_IMAGE_EMBD=$V/out448-cpu --setenv=BRECCIA_KV=f16
python3 $V/vis-cmp.py $V/ref/resp448.json out/report-vi-f16.json
arm vi-f16-1024 out/breccia-vis $V/prompt-img.ids 48 1200 --setenv=BRECCIA_IMAGE_EMBD=$V/out1024-cpu --setenv=BRECCIA_KV=f16
python3 $V/vis-cmp.py $V/ref/resp1024.json out/report-vi-f16-1024.json
echo F16 DONE
