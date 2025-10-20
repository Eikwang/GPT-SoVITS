param([string]$Path)

$data = Get-Content -Raw -Path $Path | ConvertFrom-Json
$first = $data | Select-Object -First 1
$last = $data | Select-Object -Last 1

function Stats($arr) {
  $min = ($arr | Measure-Object -Minimum).Minimum
  $max = ($arr | Measure-Object -Maximum).Maximum
  $avg = ($arr | Measure-Object -Average).Average
  return [pscustomobject]@{
    min = [math]::Round($min,2)
    max = [math]::Round($max,2)
    avg = [math]::Round($avg,2)
  }
}

$rss = $data | ForEach-Object { $_.rss_mb }
$vms = $data | ForEach-Object { $_.vms_mb }
$gpu = $data | ForEach-Object { $_.gpu_mb }

$start = [datetime]::ParseExact($first.datetime,'yyyy-MM-dd HH:mm:ss',$null)
$end = [datetime]::ParseExact($last.datetime,'yyyy-MM-dd HH:mm:ss',$null)

$rssStats = Stats $rss
$vmsStats = Stats $vms
$gpuStats = Stats $gpu

[pscustomobject]@{
  file = $Path
  count = $data.Count
  window = ($end - $start).ToString()
  rss = $rssStats
  vms = $vmsStats
  gpu = $gpuStats
  rss_drift = [math]::Round(($last.rss_mb - $first.rss_mb),2)
  gpu_drift = [math]::Round(($last.gpu_mb - $first.gpu_mb),2)
} | ConvertTo-Json -Depth 4