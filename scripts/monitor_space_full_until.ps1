param(
    [string]$SshHost = "ssh2.vast.ai",
    [int]$SshPort = 39374,
    [string]$RunId = "space_full_C_11402x20",
    [string]$Session = "semae_full_C_11402x20",
    [string]$RemoteRoot = "/workspace/SemAE",
    [string]$LocalRoot = "D:\KHDL\LuanAn\tesing",
    [int]$InstanceId = 39119375,
    [datetime]$StopAt = ([datetime]"2026-06-04T18:00:00+07:00"),
    [int]$IntervalSeconds = 300,
    [switch]$Once
)

$ErrorActionPreference = "Continue"

$LocalModelDir = Join-Path $LocalRoot "models\$RunId"
$LocalLogDir = Join-Path $LocalRoot "logs"
$MonitorLog = Join-Path $LocalLogDir "${RunId}_monitor.log"
$RemoteCheckpointDir = "$RemoteRoot/checkpoints/$RunId"
$RemoteLog = "$RemoteRoot/logs/$RunId.log"
$DownstreamRunId = "space_hasos_from_${RunId}_latest"

New-Item -ItemType Directory -Force -Path $LocalModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $LocalLogDir | Out-Null

$sshBase = @(
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=NUL",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-p", "$SshPort",
    "root@$SshHost"
)
$scpBase = @(
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=NUL",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-P", "$SshPort"
)

function Write-MonitorLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
    Add-Content -Path $MonitorLog -Value "[$stamp] $Message"
}

function Invoke-ProcessChecked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [int]$TimeoutSeconds = 300
    )
    $p = Start-Process -FilePath $FilePath -ArgumentList $Arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $LocalLogDir "${RunId}_last_stdout.tmp") `
        -RedirectStandardError (Join-Path $LocalLogDir "${RunId}_last_stderr.tmp")
    if (-not $p.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        throw "$FilePath timed out after $TimeoutSeconds seconds"
    }
    $p.Refresh()
    return $p.ExitCode
}

function Invoke-RemoteText {
    param([string]$Command, [int]$TimeoutSeconds = 120)
    $outFile = Join-Path $LocalLogDir "${RunId}_remote_stdout.tmp"
    $errFile = Join-Path $LocalLogDir "${RunId}_remote_stderr.tmp"
    $args = $sshBase + @($Command)
    $p = Start-Process -FilePath "ssh.exe" -ArgumentList $args -NoNewWindow -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    if (-not $p.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        throw "ssh timed out after $TimeoutSeconds seconds"
    }
    $p.Refresh()
    return @{
        ExitCode = $p.ExitCode
        Stdout = if (Test-Path $outFile) { Get-Content -Raw -Path $outFile } else { "" }
        Stderr = if (Test-Path $errFile) { Get-Content -Raw -Path $errFile } else { "" }
    }
}

function Sync-RemoteArtifacts {
    param([switch]$All)
    Write-MonitorLog "sync start all=$All"

    $listCmd = "find '$RemoteCheckpointDir' -maxdepth 1 -type f -name '${RunId}_*_model.pt' -printf '%f\n' 2>/dev/null | sort -V"
    $listed = Invoke-RemoteText -Command $listCmd -TimeoutSeconds 120
    $files = $listed.Stdout -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    if ($files.Count -gt 0) {
        foreach ($file in $files) {
            $localPath = Join-Path $LocalModelDir $file
            if ($All -or -not (Test-Path -LiteralPath $localPath)) {
                $scpArgs = $scpBase + @("root@${SshHost}:$RemoteCheckpointDir/$file", $localPath)
                $exit = Invoke-ProcessChecked -FilePath "scp.exe" -Arguments $scpArgs -TimeoutSeconds 900
                if ($exit -eq 0) {
                    Write-MonitorLog "pulled checkpoint $file"
                } else {
                    Write-MonitorLog "checkpoint pull failed exit=$exit file=$file"
                }
            }
        }
    } else {
        Write-MonitorLog "no remote checkpoints found yet"
    }

    $localLogPath = Join-Path $LocalLogDir "$RunId.log"
    $snapshotLog = "$RemoteRoot/logs/${RunId}.sync_snapshot.log"
    Invoke-RemoteText -Command "cp '$RemoteLog' '$snapshotLog' 2>/dev/null || true" -TimeoutSeconds 60 | Out-Null
    $scpLogArgs = $scpBase + @("root@${SshHost}:$snapshotLog", $localLogPath)
    $logExit = Invoke-ProcessChecked -FilePath "scp.exe" -Arguments $scpLogArgs -TimeoutSeconds 120
    Write-MonitorLog "pulled train log exit=$logExit"
}

function Sync-RemoteOutputs {
    param([string]$OutputRunId)
    $localOutputsDir = Join-Path $LocalRoot "outputs"
    New-Item -ItemType Directory -Force -Path $localOutputsDir | Out-Null

    $items = @(
        "$RemoteRoot/outputs/$OutputRunId",
        "$RemoteRoot/outputs/${OutputRunId}_sentiment",
        "$RemoteRoot/outputs/${OutputRunId}_report.md",
        "$RemoteRoot/outputs/${OutputRunId}_report.json",
        "$RemoteRoot/outputs/${OutputRunId}_metrics.md",
        "$RemoteRoot/outputs/${OutputRunId}_metrics.json",
        "$RemoteRoot/logs/${OutputRunId}_pipeline.log"
    )
    foreach ($item in $items) {
        $exists = Invoke-RemoteText -Command "test -e '$item'" -TimeoutSeconds 60
        if ($exists.ExitCode -ne 0) {
            Write-MonitorLog "remote output missing $item"
            continue
        }
        $dest = if ($item -like "$RemoteRoot/logs/*") { $LocalLogDir } else { $localOutputsDir }
        $scpArgs = $scpBase + @("-r", "root@${SshHost}:$item", $dest)
        $exit = Invoke-ProcessChecked -FilePath "scp.exe" -Arguments $scpArgs -TimeoutSeconds 1800
        Write-MonitorLog "pulled output item=$item exit=$exit"
    }
}

function Invoke-DownstreamPipeline {
    param([string]$Reason)
    Write-MonitorLog "downstream start reason=$Reason output_run=$DownstreamRunId"

    $remote = @"
set -euo pipefail
cd '$RemoteRoot'
MODEL=`$(find '$RemoteCheckpointDir' -maxdepth 1 -type f -name '${RunId}_*_model.pt' | sort -V | tail -1)
if [ -z "`$MODEL" ]; then
  echo "DOWNSTREAM_ERROR no checkpoint found in $RemoteCheckpointDir"
  exit 10
fi
LOG='$RemoteRoot/logs/${DownstreamRunId}_pipeline.log'
exec > >(tee -a "`$LOG") 2>&1
echo "=== downstream start: `$(date -Iseconds) ==="
echo "reason=$Reason"
echo "model=`$MODEL"
echo "output_run=$DownstreamRunId"
test -f '$RemoteRoot/data/hasos/hasos_summ.json'
test -f '$RemoteRoot/data/sentencepiece/space_unigram_32k.model'
test -d '$RemoteRoot/data/seeds_hasos'
test -f '$RemoteRoot/data/hasos/aspect_taxonomy.tsv'
test -f '$RemoteRoot/data/hasos/aspect_taxonomy.json'
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
python scripts/run_space_hasos_aspect_parallel.py \
  --model "`$MODEL" \
  --run_id '$DownstreamRunId' \
  --num_shards 4 \
  --gpu 0 \
  --max_tokens 40 \
  --sentiment_split
python scripts/summarize_aspect_outputs.py --run_id '$DownstreamRunId'
python scripts/score_semae_run.py --run_id '$DownstreamRunId'
echo "=== downstream done: `$(date -Iseconds) ==="
"@
    $result = Invoke-RemoteText -Command $remote -TimeoutSeconds 21600
    Write-MonitorLog "downstream exit=$($result.ExitCode)"
    if ($result.ExitCode -eq 0) {
        Sync-RemoteOutputs -OutputRunId $DownstreamRunId
        return $true
    }

    Write-MonitorLog "downstream failed stdout_tail=$($result.Stdout.Substring([Math]::Max(0, $result.Stdout.Length - 1000))) stderr=$($result.Stderr.Trim())"
    Sync-RemoteArtifacts -All
    Sync-RemoteOutputs -OutputRunId $DownstreamRunId
    Destroy-Instance
    return $false
}

function Destroy-Instance {
    $apiKey = [Environment]::GetEnvironmentVariable("VAST_API_KEY", "Process")
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        $apiKey = [Environment]::GetEnvironmentVariable("VAST_API_KEY", "User")
    }
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        Write-MonitorLog "destroy skipped: VAST_API_KEY missing"
        return
    }
    try {
        $uri = "https://console.vast.ai/api/v0/instances/$InstanceId/?api_key=$apiKey"
        $response = Invoke-WebRequest -Method Delete -Uri $uri -UseBasicParsing -TimeoutSec 60
        Write-MonitorLog "destroy requested instance=$InstanceId status=$($response.StatusCode)"
    } catch {
        Write-MonitorLog "destroy failed instance=$InstanceId error=$($_.Exception.Message)"
    }
}

Write-MonitorLog "monitor start run=$RunId stop_at=$($StopAt.ToString('o')) instance=$InstanceId"

$completed = $false
$destroyed = $false
$onceDone = $false
while ((Get-Date) -lt $StopAt) {
    try {
        Sync-RemoteArtifacts

        $statusCmd = @"
set +e
echo STATUS_BEGIN
tmux has-session -t '$Session' 2>/dev/null; echo TMUX_EXIT=`$?
pgrep -af 'python train.py.*$RunId' >/dev/null 2>&1; echo TRAIN_EXIT=`$?
grep -q '=== done:' '$RemoteLog' 2>/dev/null; echo DONE_EXIT=`$?
grep -qE 'Traceback|RuntimeError|NONFINITE_DIAGNOSTIC|out of memory|CUDA out of memory' '$RemoteLog' 2>/dev/null; echo FAIL_EXIT=`$?
find '$RemoteCheckpointDir' -maxdepth 1 -type f -name '${RunId}_20_model.pt' | grep -q .; echo E20_EXIT=`$?
STATUS_END
"@
        $status = Invoke-RemoteText -Command $statusCmd -TimeoutSeconds 120
        $text = $status.Stdout
        Write-MonitorLog ("status " + (($text -split "`n" | Where-Object { $_ -match "_EXIT=" }) -join " "))

        $tmuxDead = $text -match "TMUX_EXIT=1"
        $trainDead = $text -match "TRAIN_EXIT=1"
        $done = $text -match "DONE_EXIT=0"
        $failed = $text -match "FAIL_EXIT=0"
        $epoch20 = $text -match "E20_EXIT=0"

        if ($epoch20 -or $done) {
            Sync-RemoteArtifacts -All
            Write-MonitorLog "training completed or epoch20 present; starting downstream"
            Invoke-DownstreamPipeline -Reason "training_complete"
            $completed = $true
            break
        }

    if ($failed -or ($tmuxDead -and $trainDead)) {
        Write-MonitorLog "failure detected failed=$failed tmux_dead=$tmuxDead train_dead=$trainDead; syncing all before destroy"
        Sync-RemoteArtifacts -All
        Destroy-Instance
        $destroyed = $true
        break
    }
    if ($Once) {
        Write-MonitorLog "once mode complete"
        $onceDone = $true
        break
    }
    } catch {
        Write-MonitorLog "monitor tick error: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $IntervalSeconds
}

if (-not $completed -and -not $destroyed -and -not $onceDone) {
    Write-MonitorLog "stop_at reached; stopping training, syncing all, and starting downstream"
    try {
        Invoke-RemoteText -Command "tmux kill-session -t '$Session' 2>/dev/null || true; pkill -f 'python train.py.*$RunId' 2>/dev/null || true" -TimeoutSeconds 120 | Out-Null
    } catch {
        Write-MonitorLog "stop training command error: $($_.Exception.Message)"
    }
    Sync-RemoteArtifacts -All
    Invoke-DownstreamPipeline -Reason "deadline"
}

Write-MonitorLog "monitor exit completed=$completed destroyed=$destroyed once=$onceDone"
