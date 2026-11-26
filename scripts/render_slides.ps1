$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$deckPath = Join-Path $projectRoot 'presentation\QuestionForge.pptx'
$previewPath = Join-Path $projectRoot 'presentation\preview'
if (Get-Process -Name POWERPNT -ErrorAction SilentlyContinue) {
    throw 'PowerPoint is already running; skipping automated rendering to preserve the existing session.'
}
New-Item -ItemType Directory -Path $previewPath -Force | Out-Null
$application = $null
$deck = $null
try {
    $application = New-Object -ComObject PowerPoint.Application
    $application.AutomationSecurity = 3
    $application.DisplayAlerts = 1
    $deck = $application.Presentations.Open($deckPath, $true, $false, $false)
    $deck.Export($previewPath, 'PNG', 1600, 900)
    $deck.SaveAs((Join-Path $projectRoot 'presentation\QuestionForge.pdf'), 32)
    Write-Output "Rendered $($deck.Slides.Count) slides into $previewPath"
}
finally {
    if ($null -ne $deck) { $deck.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($deck) }
    if ($null -ne $application) { $application.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($application) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
