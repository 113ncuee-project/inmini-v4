[CmdletBinding()]
param(
    [string]$Out = ('output/full_parallel_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [string]$Models = 'resnet18',
    [string]$Seeds = '7,37,47',
    [string]$Methods = 'rl,sa,random',
    [ValidateRange(1,4)][int]$Jobs = 2,
    [ValidateRange(6,1000000)][int]$Queries = 512,
    [ValidateRange(1,604800)][double]$Seconds = 7200,
    [ValidateRange(1,86400)][double]$EvalTimeout = 600,
    [ValidateSet('cost','residual','transition')][string]$Policy = 'cost',
    [ValidateSet('edp','energy','latency','balanced')][string]$Objective = 'edp',
    [ValidateSet('all','off')][string]$ArchitectureScreen = 'all',
    [ValidateSet('wide','off')][string]$SeedRefinement = 'wide',
    [double]$ScreenSeconds = 3600,
    [int]$SeedQueries = 48,
    [double]$SeedSeconds = 3600,
    [double]$SeedDPSeconds = 1800,
    [ValidateRange(0,1000000)][int]$ArchitectureRefreshInterval = 4,
    [ValidateRange(0,1000)][double]$EpisodeCostRise = .25,
    [ValidateRange(2,100)][int]$MaxFusionDepth = 4,
    [Nullable[double]]$MaxArea = $null,
    [Nullable[double]]$MaxPower = $null,
    [Nullable[double]]$MaxLatencyMs = $null,
    [string]$SeedCase,
    [string]$SharedCase,
    [string]$EngineApp,
    [switch]$Resume,
    [switch]$Stop,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ProjectFolder = Get-Item -LiteralPath $PSScriptRoot
while ($null -ne $ProjectFolder -and -not (Test-Path -LiteralPath (Join-Path $ProjectFolder.FullName 'inputs/replay_v04'))) {
    $ProjectFolder = $ProjectFolder.Parent
}
if ($null -eq $ProjectFolder) { throw 'Cannot locate the project inputs/replay_v04 directory.' }
$Project = $ProjectFolder.FullName
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Project $Value))
}
$Python = Join-Path $Project '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Create .venv and install simplified_rapidchiplet/requirements-replay.txt first.' }
if ([IO.Path]::IsPathRooted($Out)) { $OutputPath = [IO.Path]::GetFullPath($Out) }
else { $OutputPath = [IO.Path]::GetFullPath((Join-Path $Project $Out)) }
$OutputRoot = [IO.Path]::GetFullPath((Join-Path $Project 'output')) + [IO.Path]::DirectorySeparatorChar
if (-not $OutputPath.StartsWith($OutputRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must be a case directory inside this project output.' }
if ($Resume -and $Stop) { throw 'Choose Resume or Stop for this invocation.' }
$ProtocolPath = Join-Path $OutputPath 'cohort_protocol.json'
if ($Stop) {
    if (-not (Test-Path -LiteralPath $ProtocolPath)) { throw 'No saved concurrent experiment at this output.' }
    $StopPath = Join-Path $OutputPath 'STOP_REQUESTED'
    if ($DryRun) { @{action='request_cooperative_stop';path=$StopPath;new_PPA=0;new_SGD=0} | ConvertTo-Json; return }
    New-Item -Path $StopPath -ItemType File -Force | Out-Null
    Write-Output 'Stop requested; active evaluations checkpoint normally. Resume with the same output.'
    return
}
if ($Resume) {
    if (-not (Test-Path -LiteralPath $ProtocolPath)) { throw 'No saved concurrent experiment to resume.' }
    $Saved = Get-Content -LiteralPath $ProtocolPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $Runner = Join-Path $OutputPath 'engine/scripts/run_parallel_full_suite.py'
    $Arguments = @('-u','-X','utf8',$Runner,'--out',$OutputPath,'--resume')
    $Description = @{action='resume_saved_protocol';source_identity=$Saved.source_identity;jobs=$Saved.jobs_limit;cases=$Saved.jobs;new_PPA=0;new_SGD=0}
} else {
    if (-not $EngineApp) { $EngineApp = Join-Path $Project 'simplified_rapidchiplet' }
    $Runner = Join-Path (Resolve-ProjectPath $EngineApp) 'scripts/run_parallel_full_suite.py'
    $Culture = [Globalization.CultureInfo]::InvariantCulture
    $Arguments = @('-u','-X','utf8',$Runner,'--out',$OutputPath,'--models',$Models,'--seeds',$Seeds,'--methods',$Methods,
        '--jobs',[string]$Jobs,'--queries',[string]$Queries,'--seconds',$Seconds.ToString($Culture),
        '--eval-timeout',$EvalTimeout.ToString($Culture),'--policy',$Policy,'--objective',$Objective,
        '--architecture-screen',$ArchitectureScreen,'--screen-seconds',$ScreenSeconds.ToString($Culture),
        '--seed-refinement',$SeedRefinement,'--seed-queries',[string]$SeedQueries,'--seed-seconds',$SeedSeconds.ToString($Culture),
        '--seed-dp-seconds',$SeedDPSeconds.ToString($Culture),'--architecture-refresh-interval',[string]$ArchitectureRefreshInterval,
        '--episode-cost-rise',$EpisodeCostRise.ToString($Culture),'--max-fusion-depth',[string]$MaxFusionDepth)
    foreach ($Pair in @(@('--max-area',$MaxArea),@('--max-power',$MaxPower),@('--max-latency-ms',$MaxLatencyMs))) {
        if ($null -ne $Pair[1]) { $Arguments += @($Pair[0],$Pair[1].ToString($Culture)) }
    }
    if ($SeedCase) { $Arguments += @('--seed-case',(Resolve-ProjectPath $SeedCase)) }
    if ($SharedCase) { $Arguments += @('--shared-case',(Resolve-ProjectPath $SharedCase)) }
    $Description = @{action='fresh_concurrent_cases';jobs=$Jobs;models=$Models;seeds=$Seeds;policy=$Policy;new_PPA=0;new_SGD=0}
}
if (-not (Test-Path -LiteralPath $Runner)) { throw 'Concurrent runner missing; use a validated full-system release or its frozen output/engine.' }
if ($DryRun) {
    $Description.python = $Python
    $Description.arguments = $Arguments
    $Description.timing = 'Independent RNG cases concurrently; per-case physical oracle and all PPA costs retained. Not isolated single-search speedup.'
    $Description | ConvertTo-Json -Depth 12
    return
}
$Awake = $false
try {
    if (-not ('InminFullParallelPower' -as [type])) {
        Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class InminFullParallelPower { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
    }
    [void][InminFullParallelPower]::SetThreadExecutionState([uint32]2147483649)
    $Awake = $true
    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
} finally {
    if ($Awake) { [void][InminFullParallelPower]::SetThreadExecutionState([uint32]2147483648) }
}
exit $ExitCode
