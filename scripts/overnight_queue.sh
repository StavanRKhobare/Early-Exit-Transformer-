#!/bin/bash
# Runs the rest of the 6-hour plan unattended. Each step starts when the previous one ends. No network needed.
cd "/home/stavan-khobare/Desktop/AIML Shi" || exit 1
Q=outputs/results/queue.log
say(){ echo "$(date '+%H:%M:%S') $*" >> "$Q"; }
BLOCK2_PID=$1
say "queue started, waiting for Block 2 pid $BLOCK2_PID"
while kill -0 "$BLOCK2_PID" 2>/dev/null; do sleep 15; done
say "Block 2 process ended"
# Finishes Block 2 if the running copy died. It skips models that are already done.
if ! grep -q "BLOCK2 DONE" outputs/results/shared_backbone/progress.log; then
  say "Block 2 incomplete, resuming"; .venv/bin/python scripts/extract_heads.py >> "$Q" 2>&1 || { say "Block 2 FAILED, stopping"; exit 1; }
fi
say "Block 3 start"; .venv/bin/python scripts/certified_selection.py --repeats 50 > outputs/results/block3.log 2>&1 || { say "Block 3 FAILED, stopping"; exit 1; }
say "Block 3 done"
say "Block 4 start"; .venv/bin/python scripts/real_energy.py > outputs/results/block4.log 2>&1 && say "Block 4 done" || say "Block 4 FAILED, partial results are kept"
say "Test report start"; .venv/bin/python scripts/certified_selection.py --report-test --repeats 0 > outputs/results/block3_test.log 2>&1 && say "Test report done" || say "Test report FAILED"
say "QUEUE FINISHED"
