#!/bin/bash
# Standalone watcher that fires a macOS notification when the overnight
# experiment completes. Runs independently of VSCode / Claude Code so it
# survives session termination.
#
# Usage:
#   nohup bash experiments/notify_on_done.sh > experiments/notify.log 2>&1 &
#
# Exits when overnight_summary.json appears. Also logs periodic status.

DONE_FILE="/Users/ericdong/nmr-ssl/experiments/results_overnight/overnight_summary.json"
LOG_FILE="/Users/ericdong/nmr-ssl/experiments/notify.log"

echo "[notify] started $(date), watching $DONE_FILE"

while [ ! -f "$DONE_FILE" ]; do
  # Check every 60 seconds
  sleep 60

  # Periodic status log (every 10 minutes)
  if [ $((SECONDS % 600)) -lt 60 ]; then
    elapsed_min=$((SECONDS / 60))
    n_stage_summaries=$(ls /Users/ericdong/nmr-ssl/experiments/results_overnight/*/summary.json 2>/dev/null | wc -l | tr -d ' ')
    echo "[notify] $(date '+%Y-%m-%d %H:%M:%S') elapsed=${elapsed_min}m stages_complete=${n_stage_summaries}/5"
  fi

  # Safety: if the overnight python process is dead AND no summary file, something crashed
  if ! pgrep -f "run_overnight.py" > /dev/null 2>&1; then
    if [ ! -f "$DONE_FILE" ]; then
      echo "[notify] WARNING: run_overnight.py is no longer running but overnight_summary.json does not exist. Experiment likely crashed."
      osascript -e 'display notification "NMR experiment CRASHED — check logs" with title "nmr-ssl" sound name "Basso"' 2>/dev/null
      exit 2
    fi
  fi
done

# Success path
echo "[notify] detected completion at $(date), firing notification"
osascript -e 'display notification "NMR overnight experiment COMPLETE. Check Claude Code for results." with title "nmr-ssl" subtitle "5 stages finished" sound name "Glass"' 2>/dev/null

# Also try to open Claude Code / VSCode if not running
osascript -e 'tell application "Visual Studio Code" to activate' 2>/dev/null

echo "[notify] done"
