param(
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [Parameter(Mandatory = $true)]
    [string]$OutDir,
    [string]$Prefix = "500reqs_different_rates"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

function To-Double([string]$s) {
    return [double]::Parse($s, [System.Globalization.CultureInfo]::InvariantCulture)
}

function New-ChartObject(
    [string]$title,
    [string]$xTitle,
    [string]$y1Title,
    [string]$y2Title
) {
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = 2000
    $chart.Height = 1120
    $chart.BackColor = [System.Drawing.Color]::FromArgb(246, 248, 252)
    $chart.BorderlineColor = [System.Drawing.Color]::FromArgb(220, 225, 233)
    $chart.BorderlineWidth = 1
    $chart.BorderlineDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid
    $chart.AntiAliasing = [System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles]::All
    $chart.TextAntiAliasingQuality = [System.Windows.Forms.DataVisualization.Charting.TextAntiAliasingQuality]::High

    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea "Main"
    $area.BackColor = [System.Drawing.Color]::White
    $area.Position.Auto = $true
    $area.AxisX.Title = $xTitle
    $area.AxisX.TitleFont = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
    $area.AxisX.LabelStyle.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(236, 240, 246)
    $area.AxisX.MajorGrid.LineDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dash
    $area.AxisX.LineColor = [System.Drawing.Color]::FromArgb(170, 178, 192)
    $area.AxisX.IsMarginVisible = $false
    $area.AxisX.Interval = 5

    $area.AxisY.Title = $y1Title
    $area.AxisY.TitleFont = New-Object System.Drawing.Font("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
    $area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $area.AxisY.LabelStyle.ForeColor = [System.Drawing.Color]::FromArgb(35, 85, 135)
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(232, 237, 245)
    $area.AxisY.MajorGrid.LineDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dash
    $area.AxisY.LineColor = [System.Drawing.Color]::FromArgb(170, 178, 192)
    $area.AxisY.Minimum = 0

    if ([string]::IsNullOrWhiteSpace($y2Title)) {
        $area.AxisY2.Enabled = [System.Windows.Forms.DataVisualization.Charting.AxisEnabled]::False
    } else {
        $area.AxisY2.Enabled = [System.Windows.Forms.DataVisualization.Charting.AxisEnabled]::True
        $area.AxisY2.Title = $y2Title
        $area.AxisY2.TitleFont = New-Object System.Drawing.Font("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
        $area.AxisY2.LabelStyle.Font = New-Object System.Drawing.Font("Segoe UI", 10)
        $area.AxisY2.LabelStyle.ForeColor = [System.Drawing.Color]::FromArgb(175, 92, 20)
        $area.AxisY2.MajorGrid.Enabled = $false
        $area.AxisY2.LineColor = [System.Drawing.Color]::FromArgb(170, 178, 192)
        $area.AxisY2.Minimum = 0
    }

    [void]$chart.ChartAreas.Add($area)

    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend "Legend"
    $legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
    $legend.Alignment = [System.Drawing.StringAlignment]::Center
    $legend.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
    $legend.BackColor = [System.Drawing.Color]::Transparent
    [void]$chart.Legends.Add($legend)

    $titleObj = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $titleObj.Text = $title
    $titleObj.Font = New-Object System.Drawing.Font("Segoe UI", 18, [System.Drawing.FontStyle]::Bold)
    $titleObj.ForeColor = [System.Drawing.Color]::FromArgb(33, 41, 52)
    $titleObj.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
    [void]$chart.Titles.Add($titleObj)

    return $chart
}

function Add-LineSeries(
    [System.Windows.Forms.DataVisualization.Charting.Chart]$Chart,
    [string]$Name,
    [double[]]$X,
    [double[]]$Y,
    [System.Drawing.Color]$Color,
    [System.Windows.Forms.DataVisualization.Charting.AxisType]$YAxisType,
    [System.Windows.Forms.DataVisualization.Charting.MarkerStyle]$MarkerStyle,
    [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]$DashStyle,
    [bool]$ShowLabels = $false,
    [string]$LabelFmt = "",
    [double[]]$LabelValues = $null
) {
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $Name
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
    $series.BorderWidth = 4
    $series.Color = $Color
    $series.YAxisType = $YAxisType
    $series.MarkerStyle = $MarkerStyle
    $series.MarkerSize = 9
    $series.MarkerColor = [System.Drawing.Color]::White
    $series.MarkerBorderColor = $Color
    $series.MarkerBorderWidth = 2
    $series.BorderDashStyle = $DashStyle
    $series.IsVisibleInLegend = $true

    for ($i = 0; $i -lt $X.Length; $i++) {
        $idx = $series.Points.AddXY($X[$i], $Y[$i])
        if ($ShowLabels) {
            $lv = $Y[$i]
            if ($null -ne $LabelValues -and $LabelValues.Length -gt $i) {
                $lv = $LabelValues[$i]
            }
            $series.Points[$idx].Label = $lv.ToString($LabelFmt, [System.Globalization.CultureInfo]::InvariantCulture)
            $series.Points[$idx].LabelForeColor = $Color
            $series.Points[$idx].Font = New-Object System.Drawing.Font("Segoe UI", 8, [System.Drawing.FontStyle]::Bold)
        }
    }

    [void]$Chart.Series.Add($series)
}

if (-not (Test-Path $CsvPath)) {
    throw "CSV not found: $CsvPath"
}

$rows = Import-Csv -Path $CsvPath
if (-not $rows -or $rows.Count -eq 0) {
    throw "CSV has no data: $CsvPath"
}

$requiredCols = @(
    "req_rate",
    "Total Pages Returned",
    "Total free() Time (ms)",
    "Avg Time / Call (ms)",
    "Calls that triggered UNMAP",
    "Total Pages Unmapped",
    "Total UNMAP Time",
    "Avg UNMAP Time"
)

foreach ($c in $requiredCols) {
    if (-not ($rows[0].PSObject.Properties.Name -contains $c)) {
        throw "Missing required column: $c"
    }
}

$x = @()
$totalReturned = @()
$totalFreeTime = @()
$avgCall = @()
$unmapCalls = @()
$unmapPages = @()
$totalUnmapTime = @()
$avgUnmapTime = @()

foreach ($r in $rows) {
    $x += To-Double $r."req_rate"
    $totalReturned += To-Double $r."Total Pages Returned"
    $totalFreeTime += To-Double $r."Total free() Time (ms)"
    $avgCall += To-Double $r."Avg Time / Call (ms)"
    $unmapCalls += To-Double $r."Calls that triggered UNMAP"
    $unmapPages += To-Double $r."Total Pages Unmapped"
    $totalUnmapTime += To-Double $r."Total UNMAP Time"
    $avgUnmapTime += To-Double $r."Avg UNMAP Time"
}

$maxTotalUnmap = ($totalUnmapTime | Measure-Object -Maximum).Maximum
$maxAvgUnmap = ($avgUnmapTime | Measure-Object -Maximum).Maximum
$scaleAvgUnmap = 1.0
if ($maxAvgUnmap -gt 0) {
    $scaleAvgUnmap = [math]::Max(1.0, [math]::Round($maxTotalUnmap / $maxAvgUnmap, 0))
}
$avgUnmapScaled = @($avgUnmapTime | ForEach-Object { $_ * $scaleAvgUnmap })

$unmapFreeRatioPct = @()
for ($i = 0; $i -lt $x.Length; $i++) {
    if ($totalFreeTime[$i] -gt 0) {
        $unmapFreeRatioPct += (($totalUnmapTime[$i] / $totalFreeTime[$i]) * 100.0)
    } else {
        $unmapFreeRatioPct += 0.0
    }
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

# Figure 1: keep only returned pages + avg free time
$chart1 = New-ChartObject `
    -title "500 Requests: Returned Pages and Avg Free Time by Request Rate" `
    -xTitle "Request Rate (req/s)" `
    -y1Title "Returned Pages (pages)" `
    -y2Title "Avg Free Time / Call (ms)"

Add-LineSeries -Chart $chart1 -Name "Total Pages Returned" `
    -X $x -Y $totalReturned `
    -Color ([System.Drawing.Color]::FromArgb(33, 113, 181)) `
    -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Primary) `
    -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Circle) `
    -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid)

Add-LineSeries -Chart $chart1 -Name "Avg Free Time / Call (ms)" `
    -X $x -Y $avgCall `
    -Color ([System.Drawing.Color]::FromArgb(192, 57, 99)) `
    -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Secondary) `
    -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Diamond) `
    -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid) `
    -ShowLabels $true -LabelFmt "0.0000"

$chart1Path = Join-Path $OutDir ($Prefix + "_fig1_returned_free_time.png")
$chart1.SaveImage($chart1Path, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)

# Figure 2: keep UNMAP calls/pages + total/avg UNMAP time
$chart2 = New-ChartObject `
    -title "500 Requests: UNMAP Calls, Pages, and Time by Request Rate" `
    -xTitle "Request Rate (req/s)" `
    -y1Title "Calls / Pages" `
    -y2Title "Time (ms)"

Add-LineSeries -Chart $chart2 -Name "Calls Triggering UNMAP" `
    -X $x -Y $unmapCalls `
    -Color ([System.Drawing.Color]::FromArgb(39, 174, 96)) `
    -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Primary) `
    -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Triangle) `
    -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid)

Add-LineSeries -Chart $chart2 -Name "Total Pages Unmapped (pages)" `
    -X $x -Y $unmapPages `
    -Color ([System.Drawing.Color]::FromArgb(52, 152, 219)) `
    -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Primary) `
    -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Circle) `
    -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid)

Add-LineSeries -Chart $chart2 -Name "Total UNMAP Time (ms)" `
    -X $x -Y $totalUnmapTime `
    -Color ([System.Drawing.Color]::FromArgb(243, 156, 18)) `
    -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Secondary) `
    -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Square) `
    -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid)

if ($scaleAvgUnmap -gt 1) {
    $avgLegend = "Avg UNMAP Time (ms) x$([int]$scaleAvgUnmap) (scaled)"
    Add-LineSeries -Chart $chart2 -Name $avgLegend `
        -X $x -Y $avgUnmapScaled `
        -Color ([System.Drawing.Color]::FromArgb(142, 68, 173)) `
        -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Secondary) `
        -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Diamond) `
        -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dash) `
        -ShowLabels $true -LabelFmt "0.0000" -LabelValues $avgUnmapTime

    $note = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $note.Text = "Note: Avg UNMAP line is scaled for visibility; point labels are raw ms."
    $note.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Bottom
    $note.Font = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Italic)
    $note.ForeColor = [System.Drawing.Color]::FromArgb(90, 98, 110)
    [void]$chart2.Titles.Add($note)
} else {
    Add-LineSeries -Chart $chart2 -Name "Avg UNMAP Time (ms)" `
        -X $x -Y $avgUnmapTime `
        -Color ([System.Drawing.Color]::FromArgb(142, 68, 173)) `
        -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Secondary) `
        -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Diamond) `
        -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dash) `
        -ShowLabels $true -LabelFmt "0.0000"
}

$chart2Path = Join-Path $OutDir ($Prefix + "_fig2_unmap_calls_pages_time.png")
$chart2.SaveImage($chart2Path, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)

# Figure 3: UNMAP time share in total free time
$chart3 = New-ChartObject `
    -title "500 Requests: UNMAP Time Share in Total Free Time" `
    -xTitle "Request Rate (req/s)" `
    -y1Title "UNMAP Time / Total Free Time (%)" `
    -y2Title ""

Add-LineSeries -Chart $chart3 -Name "UNMAP Time Share (%)" `
    -X $x -Y $unmapFreeRatioPct `
    -Color ([System.Drawing.Color]::FromArgb(231, 76, 60)) `
    -YAxisType ([System.Windows.Forms.DataVisualization.Charting.AxisType]::Primary) `
    -MarkerStyle ([System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Circle) `
    -DashStyle ([System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid) `
    -ShowLabels $true -LabelFmt "0.00"

$chart3Path = Join-Path $OutDir ($Prefix + "_fig3_unmap_time_ratio_pct.png")
$chart3.SaveImage($chart3Path, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)

Write-Output "Saved chart 1: $chart1Path"
Write-Output "Saved chart 2: $chart2Path"
Write-Output "Saved chart 3: $chart3Path"
