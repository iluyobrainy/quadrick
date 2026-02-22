param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [int]$OriginalPid,
    [int]$DurationMinutes = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

Set-Location $RepoRoot

while ($true) {
    $running = Get-Process -Id $OriginalPid -ErrorAction SilentlyContinue
    if ($null -eq $running) {
        break
    }
    Start-Sleep -Seconds 5
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = "logs/phase1_ext30_${stamp}.out.log"
$errLog = "logs/phase1_ext30_${stamp}.err.log"
$startUtc = [DateTimeOffset]::UtcNow
$etaUtc = $startUtc.AddMinutes($DurationMinutes)

$newProc = Start-Process `
    -FilePath "python" `
    -ArgumentList @(
        "scripts/phase1_paper_assessment.py",
        "--duration-minutes", "$DurationMinutes",
        "--start-equity", "100",
        "--poll-seconds", "30"
    ) `
    -WorkingDirectory (Get-Location) `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru `
    -WindowStyle Hidden

$state = [ordered]@{
    pid = $newProc.Id
    started_utc = $startUtc.ToString("o")
    eta_utc = $etaUtc.ToString("o")
    stdout = $outLog
    stderr = $errLog
    command = "python scripts/phase1_paper_assessment.py --duration-minutes $DurationMinutes --start-equity 100 --poll-seconds 30"
    profile = "opportunity_fallback_v5_ext30"
}

Write-Utf8NoBom -Path "logs/phase1_6h_current.json" -Content ($state | ConvertTo-Json -Depth 4)

$meta = [ordered]@{
    triggered_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    previous_pid = $OriginalPid
    new_pid = $newProc.Id
    duration_minutes = $DurationMinutes
    stdout = $outLog
    stderr = $errLog
}

Write-Utf8NoBom -Path "logs/phase1_extension_${stamp}.json" -Content ($meta | ConvertTo-Json -Depth 4)
