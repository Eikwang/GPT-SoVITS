param([string]$Path)

if (-not (Test-Path $Path)) { Write-Error "File not found: $Path"; exit 1 }

$lines = Get-Content -Path $Path

# Regex patterns (tolerant to separators and spacing)
$rxTimestamp = [regex]"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
$rxRSS = [regex]"RSS\s*[:=]?\s*(\d+\.?\d*)\s*MB?"
$rxVMS = [regex]"VMS\s*[:=]?\s*(\d+\.?\d*)\s*MB?"
$rxUSS = [regex]"USS\s*[:=]?\s*(\d+\.?\d*)\s*MB?"
$rxGPU = [regex]"GPU\s*(allocated|alloc|Alloc)?\s*[:=]?\s*(\d+\.?\d*)\s*MB?"

$records = @()
foreach ($line in $lines) {
  $rssMatch = $rxRSS.Match($line)
  $vmsMatch = $rxVMS.Match($line)
  $ussMatch = $rxUSS.Match($line)
  $gpuMatch = $rxGPU.Match($line)
  if ($rssMatch.Success -or $vmsMatch.Success -or $ussMatch.Success -or $gpuMatch.Success) {
    $ts = $null
    $tsm = $rxTimestamp.Match($line)
    if ($tsm.Success) { $ts = $tsm.Groups[1].Value }
    $records += [pscustomobject]@{
      timestamp = $ts
      rss_mb = if ($rssMatch.Success) { [double]$rssMatch.Groups[1].Value } else { $null }
      vms_mb = if ($vmsMatch.Success) { [double]$vmsMatch.Groups[1].Value } else { $null }
      uss_mb = if ($ussMatch.Success) { [double]$ussMatch.Groups[1].Value } else { $null }
      gpu_mb = if ($gpuMatch.Success) { [double]$gpuMatch.Groups[2].Value } else { $null }
    }
  }
}

function Stats($arr) {
  $nums = $arr | Where-Object { $_ -ne $null }
  if (-not $nums -or $nums.Count -eq 0) { return $null }
  $min = ($nums | Measure-Object -Minimum).Minimum
  $max = ($nums | Measure-Object -Maximum).Maximum
  $avg = ($nums | Measure-Object -Average).Average
  return [pscustomobject]@{
    min = [math]::Round($min,2)
    max = [math]::Round($max,2)
    avg = [math]::Round($avg,2)
  }
}

$rss = $records | ForEach-Object { $_.rss_mb }
$vms = $records | ForEach-Object { $_.vms_mb }
$uss = $records | ForEach-Object { $_.uss_mb }
$gpu = $records | ForEach-Object { $_.gpu_mb }

$first = $records | Where-Object { $_.rss_mb -ne $null -or $_.gpu_mb -ne $null } | Select-Object -First 1
$last = $records | Where-Object { $_.rss_mb -ne $null -or $_.gpu_mb -ne $null } | Select-Object -Last 1

$out = [pscustomobject]@{
  file = $Path
  samples = $records.Count
  rss = Stats $rss
  vms = Stats $vms
  uss = Stats $uss
  gpu = Stats $gpu
  rss_drift = if ($first -and $last -and $first.rss_mb -ne $null -and $last.rss_mb -ne $null) { [math]::Round(($last.rss_mb - $first.rss_mb),2) } else { $null }
  gpu_drift = if ($first -and $last -and $first.gpu_mb -ne $null -and $last.gpu_mb -ne $null) { [math]::Round(($last.gpu_mb - $first.gpu_mb),2) } else { $null }
}

$out | ConvertTo-Json -Depth 4