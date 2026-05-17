<#
.SYNOPSIS
  Agentic Starter installer/upgrader (Windows native / PowerShell 5.1+ / pwsh 7+).

.DESCRIPTION
  Mirror of bootstrap.sh:
    1. Auto-detects PRODUCT_NAME from manifests, STACK, and project mode
       (root vs monorepo via projects/ convention).
    2. Asks only two operational questions:
        - Append recommended ignore entries to .gitignore? (y/N)
        - Which CLI/LLM should run INIT.md?
       It never asks about team, domain, vision, personas, or product purpose.
    3. Substitutes <PRODUCT_NAME>/<STACK> only inside starter-managed paths.
    4. Never overwrites pre-existing user instruction files.
    5. Hands off to the chosen CLI/LLM to execute INIT.md.

.EXAMPLE
  PS> .\bootstrap.ps1
  PS> .\bootstrap.ps1 -NonInteractive -Cli claude -AppendGitignore yes
#>
[CmdletBinding()]
param(
  [switch]$NonInteractive,
  [string]$Cli = "",
  [ValidateSet("yes","no","")]
  [string]$AppendGitignore = ""
)

$ErrorActionPreference = "Stop"

function Read-Safe([string]$Path) {
  if (-not (Test-Path $Path)) { return "" }
  return Get-Content $Path -Raw -ErrorAction SilentlyContinue
}

function Read-JsonField([string]$Path, [string]$Field) {
  $content = Read-Safe $Path
  if ($content -match '"' + [regex]::Escape($Field) + '"\s*:\s*"([^"]+)"') { return $Matches[1] }
  return ""
}

function Read-TomlField([string]$Path, [string]$Field) {
  $content = Read-Safe $Path
  if ($content -match '(?m)^\s*' + [regex]::Escape($Field) + '\s*=\s*"([^"]+)"') { return $Matches[1] }
  return ""
}

function Read-YamlField([string]$Path, [string]$Field) {
  $content = Read-Safe $Path
  if ($content -match '(?m)^\s*' + [regex]::Escape($Field) + '\s*:\s*"?([^"\s#]+)"?') { return $Matches[1] }
  return ""
}

function Detect-Stack([string]$BaseDir = ".") {
  if (Test-Path (Join-Path $BaseDir "angular.json")) { return "angular" }
  if (Test-Path (Join-Path $BaseDir "package.json")) {
    $pkg = Read-Safe (Join-Path $BaseDir "package.json")
    if     ($pkg -match '"next"\s*:') { return "next-ts" }
    elseif ($pkg -match '"@angular/core"\s*:') { return "angular" }
    elseif ($pkg -match '"react"\s*:') { return "react-ts" }
    elseif ($pkg -match '"vue"\s*:') { return "vue-ts" }
    elseif ($pkg -match '"@nestjs/core"|"nestjs"\s*:') { return "nestjs" }
    elseif ($pkg -match '"express"\s*:') { return "node-express" }
    else { return "node-ts" }
  }
  if (Get-ChildItem -Path $BaseDir -Filter "*.csproj" -File -ErrorAction SilentlyContinue) { return "dotnet" }
  if (Get-ChildItem -Path $BaseDir -Filter "*.sln" -File -ErrorAction SilentlyContinue) { return "dotnet" }
  if ((Test-Path (Join-Path $BaseDir "pyproject.toml")) -or (Test-Path (Join-Path $BaseDir "requirements.txt"))) {
    $py = (Read-Safe (Join-Path $BaseDir "pyproject.toml")) + (Read-Safe (Join-Path $BaseDir "requirements.txt"))
    if     ($py -match '(?i)django') { return "python-django" }
    elseif ($py -match '(?i)fastapi') { return "python-fastapi" }
    elseif ($py -match '(?i)flask') { return "python-flask" }
    else { return "python" }
  }
  if (Test-Path (Join-Path $BaseDir "go.mod")) { return "go" }
  if (Test-Path (Join-Path $BaseDir "Cargo.toml")) { return "rust" }
  if (Test-Path (Join-Path $BaseDir "pubspec.yaml")) { return "flutter" }
  if (Test-Path (Join-Path $BaseDir "composer.json")) {
    if ((Read-Safe (Join-Path $BaseDir "composer.json")) -match "laravel/framework") { return "laravel" }
    return "php"
  }
  if (Test-Path (Join-Path $BaseDir "Gemfile")) { return "ruby" }
  if (Test-Path (Join-Path $BaseDir "mix.exs")) { return "elixir" }
  if (Test-Path (Join-Path $BaseDir "build.gradle.kts")) { return "kotlin-gradle" }
  if (Test-Path (Join-Path $BaseDir "build.gradle")) { return "java-gradle" }
  if (Test-Path (Join-Path $BaseDir "pom.xml")) { return "java-maven" }
  return "unknown"
}

function Detect-ProductName([string]$BaseDir = ".") {
  $name = Read-JsonField (Join-Path $BaseDir "package.json") "name"
  if ($name) { return $name }

  $angular = Read-Safe (Join-Path $BaseDir "angular.json")
  if ($angular -match '"projects"\s*:\s*\{\s*"([A-Za-z0-9_-]+)"') { return $Matches[1] }

  $csproj = Get-ChildItem -Path $BaseDir -Filter "*.csproj" -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($csproj) { return [System.IO.Path]::GetFileNameWithoutExtension($csproj.Name) }

  $name = Read-TomlField (Join-Path $BaseDir "pyproject.toml") "name"
  if ($name) { return $name }

  $name = Read-TomlField (Join-Path $BaseDir "Cargo.toml") "name"
  if ($name) { return $name }

  $name = Read-YamlField (Join-Path $BaseDir "pubspec.yaml") "name"
  if ($name) { return $name }

  $name = Read-JsonField (Join-Path $BaseDir "composer.json") "name"
  if ($name) { return ($name -split '/')[-1] }

  $goMod = Read-Safe (Join-Path $BaseDir "go.mod")
  if ($goMod -match '(?m)^module\s+(.+)$') { return ($Matches[1].Trim() -split '/')[-1] }

  return (Get-Item $BaseDir).Name
}

function Detect-ProjectMode {
  if (-not (Test-Path "projects" -PathType Container)) { return "root" }
  $projects = Get-ChildItem "projects" -Directory -ErrorAction SilentlyContinue | Where-Object { -not $_.Name.StartsWith(".") }
  if ($projects.Count -gt 0) { return "monorepo" }
  return "root"
}

function Detect-Projects {
  if (-not (Test-Path "projects" -PathType Container)) { return @() }
  return Get-ChildItem "projects" -Directory -ErrorAction SilentlyContinue |
    Where-Object { -not $_.Name.StartsWith(".") } |
    Sort-Object Name |
    ForEach-Object {
      [ordered]@{
        name = Detect-ProductName $_.FullName
        path = ("projects/" + $_.Name)
        stack = Detect-Stack $_.FullName
      }
    }
}

$ProjectMode = Detect-ProjectMode
$Projects = Detect-Projects
$ProductName = if ($ProjectMode -eq "monorepo") { (Get-Item ".").Name } else { Detect-ProductName "." }
$Stack = if ($ProjectMode -eq "monorepo") { "monorepo" } else { Detect-Stack "." }

Write-Host "=========================================="
Write-Host "  Agentic Starter - Bootstrap (PowerShell)"
Write-Host "=========================================="
Write-Host ""
Write-Host "Auto-detected (agent will infer team/domain/personas/vision from code):"
Write-Host "  PROJECT_MODE: $ProjectMode"
Write-Host "  PRODUCT_NAME: $ProductName"
Write-Host "  STACK:        $Stack"
if ($ProjectMode -eq "monorepo") {
  Write-Host ("  PROJECTS:     " + (($Projects | ConvertTo-Json -Compress -Depth 5)))
}
Write-Host ""

$ProtectedInstructionFiles = @("AGENTS.md", "CLAUDE.md", "INIT.md", ".github/copilot-instructions.md")
$ExistingInstructionFiles = @()
foreach ($file in $ProtectedInstructionFiles) {
  if (Test-Path $file) {
    $content = Read-Safe $file
    if ($content -and $content -notmatch 'Agentic Starter|<PRODUCT_NAME>|<STACK>') {
      $ExistingInstructionFiles += $file
    }
  }
}

if ($ExistingInstructionFiles.Count -gt 0) {
  Write-Host "Detected pre-existing instruction files (will be preserved):"
  foreach ($file in $ExistingInstructionFiles) { Write-Host "  - $file" }
  Write-Host "  -> INIT.md will READ them and IMPROVE in place (essence preserved)."
  Write-Host ""
}

$StarterDirs = @(".specs", ".agents", ".skills", ".claude", ".codex")
$StarterGithubPatterns = @(
  ".github/copilot-instructions.md",
  ".github/copilot",
  ".github/PULL_REQUEST_TEMPLATE.md",
  ".github/ISSUE_TEMPLATE",
  ".github/workflows/ci.yml",
  ".github/workflows/dod.yml"
)
$StarterRootFiles = @(
  "AGENTS.md", "CLAUDE.md", "INIT.md", "_BOOTSTRAP.md",
  "README.md", "README.pt-BR.md", "playwright.config.ts"
)
$Touched = 0

function Substitute-InFile([string]$Path) {
  if (-not (Test-Path $Path -PathType Leaf)) { return }
  try {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -eq 0) { return }
    $head = if ($bytes.Length -gt 8192) { $bytes[0..8191] } else { $bytes }
    if ($head -contains 0) { return }
    $content = [System.Text.Encoding]::UTF8.GetString($bytes)
    if ($content -notmatch '<PRODUCT_NAME>|<STACK>') { return }
    $next = $content.Replace("<PRODUCT_NAME>", $script:ProductName).Replace("<STACK>", $script:Stack)
    if ($next -ne $content) {
      [System.IO.File]::WriteAllText($Path, $next, [System.Text.UTF8Encoding]::new($false))
      $script:Touched++
    }
  } catch {
  }
}

Write-Host "Substituting placeholders inside starter-managed paths..."
foreach ($dir in $StarterDirs) {
  if (Test-Path $dir -PathType Container) {
    Get-ChildItem -Path $dir -Recurse -File -Include @("*.md","*.json","*.toml","*.yml","*.yaml","*.ts") -ErrorAction SilentlyContinue |
      ForEach-Object { Substitute-InFile $_.FullName }
  }
}
foreach ($item in $StarterGithubPatterns) {
  if (Test-Path $item -PathType Container) {
    Get-ChildItem -Path $item -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object { Substitute-InFile $_.FullName }
  } elseif (Test-Path $item -PathType Leaf) {
    Substitute-InFile $item
  }
}
foreach ($file in $StarterRootFiles) { Substitute-InFile $file }
Write-Host "-> $Touched files updated (only starter-managed paths)."
Write-Host ""

$RecommendedIgnores = @"
# === Agentic Starter (auto-managed) - do not remove this header ===
# Local agent state and ephemeral artifacts created by the starter.
.starter-meta.json
.codex/local
.codex/history
.claude/sessions
.claude/cache

# Test artifacts (Playwright + coverage)
test-results/
playwright-report/
playwright/.cache/
coverage/
.nyc_output/

# Env vars
.env
.env.*
!.env.example

# Editor / OS
.DS_Store
Thumbs.db
*.swp
*.swo

# Build / dist (most common)
node_modules/
dist/
build/
out/
.next/
.nuxt/
.turbo/
.vercel/
*.tsbuildinfo

# Logs
*.log
npm-debug.log*
yarn-debug.log*
pnpm-debug.log*

# Tarballs
*.tgz
*.tar.gz
"@

$GitattributesContent = @"
# Cross-platform line endings.
* text=auto eol=lf

# Shell scripts MUST be LF.
*.sh        text eol=lf
*.bash      text eol=lf

# Windows scripts MUST be CRLF.
*.ps1       text eol=crlf
*.psm1      text eol=crlf
*.psd1      text eol=crlf
*.bat       text eol=crlf
*.cmd       text eol=crlf

# Common config / source.
*.md        text
*.json      text
*.jsonc     text
*.yml       text
*.yaml      text
*.toml      text
*.xml       text
*.html      text
*.css       text
*.scss      text
*.js        text
*.jsx       text
*.ts        text
*.tsx       text
*.mjs       text
*.cjs       text
*.py        text
*.cs        text
*.csproj    text
*.sln       text eol=crlf
*.go        text
*.rs        text
*.java      text
*.kt        text
*.kts       text
*.gradle    text
"@

function Handle-Gitignore {
  $choice = $script:AppendGitignore
  if ([string]::IsNullOrEmpty($choice) -and -not $script:NonInteractive) {
    Write-Host "=========================================="
    Write-Host "  .gitignore"
    Write-Host "=========================================="
    if (Test-Path ".gitignore") {
      Write-Host "An existing .gitignore was found."
      Write-Host "I can APPEND recommended entries (your existing content is NEVER modified)."
    } else {
      Write-Host "No .gitignore found. I can CREATE one with recommended entries."
    }
    $response = Read-Host "Proceed? [y/N]"
    if (-not $response) { $response = "n" }
    $choice = if ($response.Substring(0,1).ToLower() -in @("y","s")) { "yes" } else { "no" }
    Write-Host ""
  }
  if (-not $choice) { $choice = "no" }

  if ($choice -ne "yes") {
    Write-Host "-> .gitignore left untouched."
    return
  }

  if (Test-Path ".gitignore") {
    $existing = Read-Safe ".gitignore"
    if ($existing -match "Agentic Starter \(auto-managed\)") {
      Write-Host "-> Recommended entries already present in .gitignore. Nothing to do."
    } else {
      Add-Content -Path ".gitignore" -Value "`n$RecommendedIgnores"
      Write-Host "-> Recommended entries APPENDED to .gitignore (original content preserved)."
    }
  } else {
    Set-Content -Path ".gitignore" -Value $RecommendedIgnores -Encoding UTF8
    Write-Host "-> .gitignore CREATED."
  }

  if (-not (Test-Path ".gitattributes")) {
    Set-Content -Path ".gitattributes" -Value $GitattributesContent -Encoding UTF8
    Write-Host "-> .gitattributes CREATED."
  } else {
    Write-Host "-> .gitattributes left untouched (already exists)."
  }
}

Handle-Gitignore
Write-Host ""

$meta = [ordered]@{
  product_name = $ProductName
  stack = $Stack
  project_mode = $ProjectMode
  projects = $Projects
  bootstrapped_at = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  starter_version = "0.1.20"
  existing_instruction_files = $ExistingInstructionFiles
  init_must_ask = @()
  init_must_infer = @("team","domain","vision_oneliner","personas_beyond_dev")
  default_persona = "developer"
  init_must_merge = $ExistingInstructionFiles
  read_only_globs = @(
    "**/*.razor","**/*.cs","**/*.csproj","**/*.sln","package.json",
    "pnpm-lock.yaml","yarn.lock","package-lock.json","**/*.py","**/*.go",
    "**/*.rs","**/*.java","**/*.kt","**/*.dart","**/*.php","**/*.rb"
  )
}
$meta | ConvertTo-Json -Depth 6 | Out-File -FilePath ".starter-meta.json" -Encoding utf8
Write-Host "-> .starter-meta.json saved."
Write-Host ""

$InitPrompt = 'Read INIT.md and execute it. Do NOT modify any user source files (.razor, .cs, .ts, .py, .go, .rs, package.json, etc). Only write inside .specs/, .agents/, .skills/, .claude/, .codex/, .github/copilot*, .github/workflows/dod.yml plus root AGENTS.md/CLAUDE.md/INIT.md/README*.md. If AGENTS.md/CLAUDE.md/copilot-instructions.md already existed before bootstrap (see .starter-meta.json), READ them and IMPROVE in place - preserve their essence. DO NOT ask the human about team, domain, vision, personas, or product purpose: infer ALL of them by reading the codebase (README, package.json/angular.json/*.csproj/pyproject.toml/etc, entry points, routes, tests, env.example). Default persona is "developer"; additional personas must be derived from code (auth roles, route guards, UI flows, customer-facing copy). Honor projects/ convention: if .starter-meta.json.project_mode == "monorepo", iterate over .starter-meta.json.projects[] and produce per-project .specs/. Use parallel multi-agents.'

$CliOpts = @(
  @{ Key="claude"; Label="Claude Code"; Cmd="claude" },
  @{ Key="codex"; Label="Codex CLI"; Cmd="codex" },
  @{ Key="cursor"; Label="Cursor Agent (cursor-agent)"; Cmd="cursor-agent" },
  @{ Key="vscode"; Label="VS Code Agent Mode (paste into Chat)"; Cmd="code" },
  @{ Key="windsurf"; Label="Windsurf / Cascade (Codeium)"; Cmd="windsurf" },
  @{ Key="kiro"; Label="Kiro (AWS, paste into Chat)"; Cmd="kiro" },
  @{ Key="copilot"; Label="GitHub Copilot CLI (chat - no agent loop)"; Cmd="gh" },
  @{ Key="deepseek"; Label="Deepseek (via aider --model deepseek/deepseek-coder)"; Cmd="aider" },
  @{ Key="kimi"; Label="Kimi K2.6 (via aider --model openrouter/moonshotai/kimi-k2)"; Cmd="aider" },
  @{ Key="minimax"; Label="MiniMax M2.7 (via aider --model openrouter/minimax/minimax-text-01)"; Cmd="aider" },
  @{ Key="glm"; Label="GLM 5.1 (via aider --model openrouter/z-ai/glm-4.5)"; Cmd="aider" },
  @{ Key="hermes"; Label="Hermes Agent (Nous Research)"; Cmd="hermes" },
  @{ Key="openclaw"; Label="OpenClaw"; Cmd="openclaw" },
  @{ Key="aider"; Label="Aider (pick model interactively)"; Cmd="aider" },
  @{ Key="other"; Label="Other / manual (copy prompt to clipboard)"; Cmd="" },
  @{ Key="skip"; Label="Skip - I will run INIT.md later"; Cmd="" }
)

function Has-Cmd([string]$Name) {
  if ([string]::IsNullOrEmpty($Name)) { return $false }
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Copy-ToClipboard([string]$Text) {
  try { Set-Clipboard -Value $Text; return $true } catch { return $false }
}

function Choose-Cli {
  if (-not [string]::IsNullOrEmpty($script:Cli)) { return $script:Cli }
  if ($script:NonInteractive) { return "skip" }

  Write-Host "=========================================="
  Write-Host "  Choose the CLI/LLM to run INIT.md"
  Write-Host "=========================================="
  Write-Host ""
  for ($i = 0; $i -lt $CliOpts.Count; $i++) {
    $opt = $CliOpts[$i]
    $mark = if (Has-Cmd $opt.Cmd) { "  [installed]" } else { "" }
    Write-Host ("  [{0,2}] {1}{2}" -f ($i + 1), $opt.Label, $mark)
  }
  Write-Host ""
  $response = Read-Host ("Number [{0}]" -f $CliOpts.Count)
  if (-not $response) { $response = [string]$CliOpts.Count }
  $index = 0
  if (-not [int]::TryParse($response, [ref]$index)) { $index = $CliOpts.Count }
  if ($index -lt 1 -or $index -gt $CliOpts.Count) { $index = $CliOpts.Count }
  return $CliOpts[$index - 1].Key
}

function Require-Cmd([string]$Name, [string]$Hint) {
  if (-not (Has-Cmd $Name)) {
    Write-Host "$Name not installed: $Hint"
    exit 1
  }
}

$CliChoice = Choose-Cli

Write-Host ""
Write-Host "=========================================="
Write-Host "  Handing off to: $CliChoice"
Write-Host "=========================================="
Write-Host ""

switch ($CliChoice) {
  "claude" {
    Require-Cmd "claude" "https://docs.claude.com/claude-code"
    & claude $InitPrompt
    exit $LASTEXITCODE
  }
  "codex" {
    Require-Cmd "codex" "https://github.com/openai/codex"
    & codex exec $InitPrompt
    exit $LASTEXITCODE
  }
  "cursor" {
    Require-Cmd "cursor-agent" "Cursor 3.0+ required"
    & cursor-agent $InitPrompt
    exit $LASTEXITCODE
  }
  "vscode" {
    if (Copy-ToClipboard $InitPrompt) { Write-Host "Prompt copied to clipboard." } else { Write-Host "(clipboard unavailable - copy manually below)" }
    Write-Host ""
    Write-Host "VS Code Agent Mode runs in-IDE."
    Write-Host "1) Open this folder in VS Code."
    Write-Host "2) Open Chat and switch to Agent mode."
    Write-Host "3) Paste the prompt below:"
    Write-Host ""
    Write-Host "  $InitPrompt"
    if (Has-Cmd "code") { & code . | Out-Null }
  }
  "windsurf" {
    if (Copy-ToClipboard $InitPrompt) { Write-Host "Prompt copied to clipboard." } else { Write-Host "(clipboard unavailable - copy manually below)" }
    Write-Host ""
    Write-Host "Windsurf Cascade runs in-IDE."
    Write-Host "1) Open this folder in Windsurf."
    Write-Host "2) Open Cascade in Write mode."
    Write-Host "3) Paste the prompt below:"
    Write-Host ""
    Write-Host "  $InitPrompt"
    if (Has-Cmd "windsurf") { & windsurf . | Out-Null }
  }
  "kiro" {
    if (Copy-ToClipboard $InitPrompt) { Write-Host "Prompt copied to clipboard." } else { Write-Host "(clipboard unavailable - copy manually below)" }
    Write-Host ""
    Write-Host "Kiro runs in-IDE."
    Write-Host "1) Open this folder in Kiro."
    Write-Host "2) Open Chat in Agent mode."
    Write-Host "3) Paste the prompt below:"
    Write-Host ""
    Write-Host "  $InitPrompt"
    if (Has-Cmd "kiro") { & kiro . | Out-Null }
  }
  "copilot" {
    Require-Cmd "gh" "https://cli.github.com"
    if (Copy-ToClipboard $InitPrompt) { Write-Host "Prompt copied to clipboard." } else { Write-Host "(clipboard unavailable - copy manually below)" }
    Write-Host ""
    Write-Host "GitHub Copilot CLI does not run an autonomous agent loop."
    Write-Host "Open Copilot Chat (VS Code / IDE) and paste the prompt:"
    Write-Host ""
    Write-Host "  $InitPrompt"
  }
  "deepseek" {
    Require-Cmd "aider" "pipx install aider-chat"
    & aider --model deepseek/deepseek-coder --message $InitPrompt
    exit $LASTEXITCODE
  }
  "kimi" {
    Require-Cmd "aider" "pipx install aider-chat"
    & aider --model openrouter/moonshotai/kimi-k2 --message $InitPrompt
    exit $LASTEXITCODE
  }
  "minimax" {
    Require-Cmd "aider" "pipx install aider-chat"
    & aider --model openrouter/minimax/minimax-text-01 --message $InitPrompt
    exit $LASTEXITCODE
  }
  "glm" {
    Require-Cmd "aider" "pipx install aider-chat"
    & aider --model openrouter/z-ai/glm-4.5 --message $InitPrompt
    exit $LASTEXITCODE
  }
  "hermes" {
    Require-Cmd "hermes" "https://github.com/NousResearch/hermes-agent"
    Copy-ToClipboard $InitPrompt | Out-Null
    Write-Host "(prompt copied to clipboard as fallback)"
    & hermes $InitPrompt
    exit $LASTEXITCODE
  }
  "openclaw" {
    Require-Cmd "openclaw" "npm install -g openclaw@latest"
    Copy-ToClipboard $InitPrompt | Out-Null
    Write-Host "(prompt copied to clipboard as fallback)"
    & openclaw $InitPrompt
    exit $LASTEXITCODE
  }
  "aider" {
    Require-Cmd "aider" "pipx install aider-chat"
    & aider --message $InitPrompt
    exit $LASTEXITCODE
  }
  "other" {
    if (Copy-ToClipboard $InitPrompt) {
      Write-Host "Prompt copied to clipboard. Paste it into your CLI/agent of choice."
    } else {
      Write-Host "(clipboard unavailable - copy the prompt below manually)"
    }
    Write-Host ""
    Write-Host "Prompt:"
    Write-Host "  $InitPrompt"
  }
  default {
    @"
Skipped CLI handoff. To run INIT.md later, open your agent and paste:

  $InitPrompt

Recommended next steps:
  1) Open an agent in this folder.
  2) Paste the prompt above.
  3) Review .specs/product/VISION.md, DOMAIN.md, architecture/DESIGN.md.
  4) git add -A && git commit -m "chore: bootstrap agentic starter"

Docs: https://github.com/wesleysimplicio/agentic-starter
"@ | Write-Host
  }
}
