[CmdletBinding()]
param(
    [string[]]$Models = @('resnet18'),
    [string[]]$Seeds = @('7', '17', '37'),
    [string]$Methods = 'rl,sa,random',
    [ValidateRange(8,1000000)][int]$Queries = 512,
    [ValidateRange(1,604800)][int]$SecondsPerMethod = 3600,
    [ValidateRange(1,86400)][int]$EvaluationTimeout = 600,
    [ValidateRange(1,86400)][int]$ScreenSeconds = 3600,
    [ValidateRange(2,256)][int]$FusionDepth = 4,
    [ValidateSet('balanced','edp','latency','energy')][string]$Objective = 'edp',
    [ValidateSet('all','off')][string]$ArchitectureScreen = 'all',
    [ValidateSet('cost','transition')][string]$Policy = 'cost',
    [ValidateRange(0,1000)][double]$EpisodeCostRise = 0.25,
    [ValidateRange(0,1)][double]$EpisodeRestartProbability = 0.9,
    [switch]$DisableCostRestart,
    [string]$OutputRoot = 'output/full_system_overnight',
    [switch]$Resume,
    [switch]$KeepAwake,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$Models = @($Models | ForEach-Object { $_.Split(',') } | ForEach-Object { $_.Trim() })
$Seeds = @($Seeds | ForEach-Object { $_.Split(',') } | ForEach-Object { [int]$_.Trim() })
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Runner = Join-Path $ProjectRoot 'simplified_rapidchiplet\full_system_dse.py'
$OutputPath = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputRoot))
if (-not $OutputPath.StartsWith($ProjectRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'OutputRoot must be inside project.' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Create .venv and install simplified_rapidchiplet/requirements-replay.txt first.' }
function Get-FullSourceIdentity {
    $scripts = Join-Path $ProjectRoot 'simplified_rapidchiplet\scripts'
    $value = & $Python -c 'import sys; sys.path.insert(0,sys.argv[1]); from run_local_rl_diagnostics import sources,sha; print(sha(sources()))' $scripts
    if ($LASTEXITCODE -ne 0) { throw 'Could not hash sources.' }
    return ([string]$value).Trim()
}
$SourceIdentity = Get-FullSourceIdentity
$ProtocolPath = Join-Path $OutputPath 'overnight_protocol.json'
if (-not $DryRun) {
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $ProtocolPath)) { throw 'No saved overnight protocol; resume an individual case with the Python CLI.' }
        $saved = Get-Content -LiteralPath $ProtocolPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($saved.source_identity -ne $SourceIdentity) { throw 'Source cohort changed. Resume individual frozen cases with the Python CLI or use a new OutputRoot.' }
        $Models = @($saved.models); $Seeds = @($saved.seeds); $Methods = [string]$saved.methods
        $Queries = [int]$saved.queries; $SecondsPerMethod = [int]$saved.seconds
        $EvaluationTimeout = [int]$saved.oracle; $ScreenSeconds = [int]$saved.screen_seconds
        $FusionDepth = [int]$saved.fusion_depth; $Objective = [string]$saved.objective
        $ArchitectureScreen = [string]$saved.screen
        # Old saved protocols keep their original transition-Q/no-cut setup.
        if ($saved.PSObject.Properties.Name -contains 'policy') {
            $Policy=[string]$saved.policy; $EpisodeCostRise=[double]$saved.episode_cost_rise
            $EpisodeRestartProbability=[double]$saved.episode_restart_probability
            $DisableCostRestart=[bool]$saved.disable_cost_restart
        } else { $Policy='transition'; $DisableCostRestart=$true }
    } else {
        if (Test-Path -LiteralPath $ProtocolPath) { throw 'Output exists. Use -Resume or a new OutputRoot.' }
        New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
        @{source_identity=$SourceIdentity; models=$Models; seeds=$Seeds; methods=$Methods; queries=$Queries;
          seconds=$SecondsPerMethod; oracle=$EvaluationTimeout; screen_seconds=$ScreenSeconds;
          fusion_depth=$FusionDepth; objective=$Objective; screen=$ArchitectureScreen; policy=$Policy;
          episode_cost_rise=$EpisodeCostRise; episode_restart_probability=$EpisodeRestartProbability;
          disable_cost_restart=[bool]$DisableCostRestart} |
            ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ProtocolPath -Encoding UTF8
    }
}
$awake = $false
try {
    if ($KeepAwake -and -not $DryRun) {
        Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class InminFullPower { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
        [void][InminFullPower]::SetThreadExecutionState([uint32]2147483649)
        $awake = $true
    }
    foreach ($Model in $Models) {
        $SeedCase = Join-Path $ProjectRoot "inputs\replay_v04\$Model"
        if (-not (Test-Path -LiteralPath (Join-Path $SeedCase 'seed_attempts.json'))) { throw "No portable input for $Model" }
        foreach ($Seed in $Seeds) {
            if (-not $DryRun -and (Get-FullSourceIdentity) -ne $SourceIdentity) { throw 'Sources changed during the overnight cohort; next case was not started.' }
            $CaseOut = Join-Path $OutputPath "$Model\$Seed"
            $Arguments = @('-u','-X','utf8',$Runner,'--seed-case',$SeedCase,'--out',$CaseOut,'--methods',$Methods,
                '--seed',"$Seed",'--queries',"$Queries",'--seconds',"$SecondsPerMethod",'--eval-timeout',"$EvaluationTimeout",
                '--screen-seconds',"$ScreenSeconds",'--architecture-screen',$ArchitectureScreen,'--objective',$Objective,
                '--max-fusion-depth',"$FusionDepth")
            if ($Policy -eq 'cost') { $Arguments += '--cost-value-q' }
            if (-not $DisableCostRestart) {
                $Arguments += @('--episode-cost-rise',"$EpisodeCostRise",'--episode-restart-probability',"$EpisodeRestartProbability")
            }
            if ($Resume -and (Test-Path -LiteralPath (Join-Path $CaseOut 'effective_config.json'))) { $Arguments += '--resume' }
            Write-Host "Model=$Model RNG=$Seed Methods=$Methods Output=$CaseOut"
            if ($DryRun) { Write-Host ($Arguments -join ' '); continue }
            & $Python @Arguments
            if ($LASTEXITCODE -ne 0) { throw "Case stopped or failed ($LASTEXITCODE). Inspect $CaseOut\runner.log. Resume with the same OutputRoot." }
            & $Python -X utf8 (Join-Path $ProjectRoot 'simplified_rapidchiplet\scripts\analyze_full_system.py') --out $CaseOut
            if ($LASTEXITCODE -ne 0) { throw 'Evidence audit failed; following cases were not started.' }
            & $Python -X utf8 (Join-Path $ProjectRoot 'simplified_rapidchiplet\scripts\evaluate_full_policy.py') --out $CaseOut
            if ($LASTEXITCODE -ne 0) { throw 'Held-out policy evaluation failed.' }
            & $Python -X utf8 (Join-Path $ProjectRoot 'simplified_rapidchiplet\scripts\summarize_full_system_suite.py') --out $OutputPath
            if ($LASTEXITCODE -ne 0) { throw 'Suite report generation failed.' }
        }
    }
} finally {
    if ($awake) { [void][InminFullPower]::SetThreadExecutionState([uint32]2147483648) }
}
