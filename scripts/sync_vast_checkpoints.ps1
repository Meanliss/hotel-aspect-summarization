param(
    [string]$SshHost = "ssh2.vast.ai",
    [int]$SshPort = 39374,
    [string]$RunId = "space_full_11402x20",
    [string]$RemoteDir = "",
    [string]$LocalDir = "",
    [string]$LocalLogDir = "D:\KHDL\LuanAn\tesing\logs",
    [int]$IntervalSeconds = 600
)

$ErrorActionPreference = "Continue"
if ([string]::IsNullOrWhiteSpace($RemoteDir)) {
    $RemoteDir = "/workspace/SemAE/checkpoints/$RunId"
}
if ([string]::IsNullOrWhiteSpace($LocalDir)) {
    $LocalDir = "D:\KHDL\LuanAn\tesing\models\$RunId"
}
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
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
$syncLog = Join-Path $LocalLogDir "${RunId}_sync.log"

function Invoke-WithTimeout {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [int]$TimeoutSeconds = 180,
        [string]$StdoutPath = "",
        [string]$StderrPath = ""
    )

    $startArgs = @{
        FilePath = $FilePath
        ArgumentList = $Arguments
        NoNewWindow = $true
        PassThru = $true
    }
    if ($StdoutPath) { $startArgs.RedirectStandardOutput = $StdoutPath }
    if ($StderrPath) { $startArgs.RedirectStandardError = $StderrPath }

    $process = Start-Process @startArgs
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "$FilePath timed out after $TimeoutSeconds seconds"
    }
    return $process.ExitCode
}

while ($true) {
    $stamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
    Add-Content -Path $syncLog -Value "[$stamp] sync tick"

    try {
        $files = @("${RunId}_4pkm_model.pt")
        $files += 1..20 | ForEach-Object { "${RunId}_${_}_model.pt" }
        foreach ($file in $files) {
            if ([string]::IsNullOrWhiteSpace($file)) { continue }
            $localPath = Join-Path $LocalDir $file
            if (-not (Test-Path -LiteralPath $localPath)) {
                $scpArgs = $scpBase + @("root@${SshHost}:$RemoteDir/$file", $localPath)
                $exitCode = Invoke-WithTimeout -FilePath "scp.exe" -Arguments $scpArgs -TimeoutSeconds 600
                if ($exitCode -eq 0) {
                    Add-Content -Path $syncLog -Value "[$stamp] pulled $file"
                }
            }
        }

        if (Test-Path -LiteralPath (Join-Path $LocalDir "${RunId}_20_model.pt")) {
            Add-Content -Path $syncLog -Value "[$stamp] epoch 20 is local; final checkpoint sync complete"
            break
        }
    } catch {
        Add-Content -Path $syncLog -Value "[$stamp] sync error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $IntervalSeconds
}
