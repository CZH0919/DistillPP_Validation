# Run locally through an approved SSH execution. Text/core artifacts only.
$ErrorActionPreference = 'Stop'
$projectRoot = 'D:\Research\PK_8H200\DistillPP_Validation'
$payload = ssh.exe -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 h200 'timeout 60s /data/chenjiayu/miniconda3/envs/changzihe/bin/python -B /data/chenjiayu/changzihe/DistillPP_Validation/tools/collect_s2r_core.py'
$sshExit = $LASTEXITCODE
if ($sshExit -ne 0) { throw "Core collection failed, SSH exit=$sshExit" }
$package = ($payload -join "`n") | ConvertFrom-Json
if ($package.status -ne 'complete_core_only' -or $package.total_bytes -gt 32MB) { throw 'Invalid core package' }
$pending = @()
foreach ($entry in $package.files) {
    if ($entry.path -notmatch '^(Result/S2R_20261017/|runs/s2r_20261017/)' -or $entry.path -match '(^|/)\.\.(/|$)' -or $entry.path -notmatch '\.(json|csv|md)$') { throw 'Unexpected core path' }
    $target = [IO.Path]::GetFullPath((Join-Path $projectRoot $entry.path))
    if (-not $target.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escapes workspace' }
    $bytes = [Convert]::FromBase64String($entry.data)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $actual = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-','').ToLowerInvariant() } finally { $sha.Dispose() }
    if ($bytes.Length -ne $entry.bytes -or $actual -ne $entry.sha256) { throw "Payload mismatch: $target" }
    if (Test-Path -LiteralPath $target) {
        if ((Get-Item -LiteralPath $target).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refuse linked destination' }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $actual) { throw "Different existing core artifact: $target" }
    }
    $pending += [pscustomobject]@{target=$target;bytes=$bytes;sha256=$actual;relative=$entry.path}
}
foreach ($entry in $pending) {
    $directory = [IO.Path]::GetDirectoryName($entry.target)
    $cursor = $directory
    while ($cursor.Length -ge $projectRoot.Length) {
        if ((Test-Path -LiteralPath $cursor) -and ((Get-Item -LiteralPath $cursor).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Refuse linked parent' }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    if (-not (Test-Path -LiteralPath $entry.target)) {
        $stream = [IO.File]::Open($entry.target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
        try { $stream.Write($entry.bytes,0,$entry.bytes.Length) } finally { $stream.Dispose() }
    }
    if ((Get-FileHash -LiteralPath $entry.target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Written core hash mismatch' }
}
$receipt = [ordered]@{status='synced_verified';total_bytes=$package.total_bytes;files=@($pending | Select-Object relative,sha256)}
$receiptPath = Join-Path $projectRoot 'Project_Progress\Stage_Progress\S2R_CORE_SYNC.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 8) + "`n"), [Text.UTF8Encoding]::new($false))
Write-Output "S2R_CORE_SYNC_OK files=$($pending.Count) bytes=$($package.total_bytes)"
