param(
    [string]$SshHost = "ssh2.vast.ai",
    [int]$SshPort = 39374,
    [string]$RunId = "space_full_11402x20_stable",
    [string]$RemoteRoot = "/workspace/SemAE"
)

$ErrorActionPreference = "Stop"

$remote = @"
set -e
LOG="$RemoteRoot/logs/$RunId.log"
CKPT="$RemoteRoot/checkpoints/$RunId"
echo "=== gpu ==="
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo "=== train process ==="
ps -eo pid,ppid,pcpu,pmem,etime,cmd | grep '[p]ython train.py' || true
echo "=== tmux ==="
tmux ls 2>/dev/null || true
echo "=== checkpoints ==="
find "`$CKPT" -maxdepth 1 -type f -name "${RunId}_*_model.pt" -printf "%f %s bytes %TY-%Tm-%Td %TH:%TM:%TS\n" 2>/dev/null | sort -V | tail -25 || true
echo "=== log summary ==="
python - <<'PY'
from pathlib import Path
run_id = "$RunId"
log = Path("$RemoteRoot") / "logs" / f"{run_id}.log"
if not log.exists():
    print("missing log", log)
    raise SystemExit
s = log.read_text(errors="ignore")
marker = "=== SPACE full 11402x20 start:"
if marker in s:
    s = marker + s.split(marker)[-1]
print("skip_grad_count", s.count("Skipping non-finite grad norm"))
print("skip_loss_count", s.count("Skipping non-finite loss"))
print("nonfinite_diagnostic_count", s.count("NONFINITE_DIAGNOSTIC"))
print("epoch_diagnostic_count", s.count("EPOCH_DIAGNOSTIC"))
print("oom_count", s.lower().count("out of memory"))
print("nan_word_count", s.lower().count("nan"))
print("inf_word_count", s.lower().count("inf"))
for line in s.splitlines():
    if line.startswith("EPOCH_DIAGNOSTIC"):
        print(line)
PY
echo "=== log tail ==="
tail -40 "`$LOG" | sed -r 's/\r/\n/g' | tail -40
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
& ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $SshPort "root@$SshHost" "echo $encoded | base64 -d | bash"
