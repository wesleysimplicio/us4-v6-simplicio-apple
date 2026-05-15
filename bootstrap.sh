#!/usr/bin/env bash
# bootstrap.sh — Agentic Starter installer/upgrader.
#
# Philosophy: zero questions about the project itself. The agent reads code
# and infers domain, personas, vision, team. The only human input is operational
# (gitignore append y/N, which CLI to hand off to).
#
# What it does:
#   1. Auto-detects PRODUCT_NAME from (in order):
#        package.json "name", angular.json "projects" key, *.csproj filename,
#        pyproject.toml [project].name, Cargo.toml [package].name,
#        pubspec.yaml "name", composer.json "name", go.mod module,
#        projects/<single-subfolder> name, or cwd basename.
#      Auto-detects STACK (Node, .NET, Python, Go, Rust, Flutter, PHP, Ruby,
#      Kotlin, Java, Elixir, Angular).
#      Auto-detects PROJECT_MODE via projects/ folder:
#        - projects/ missing or empty (only .gitkeep) -> "root"  (single project at repo root)
#        - projects/ has subfolders                   -> "monorepo" (each subfolder = project)
#   2. Asks ONLY TWO operational questions:
#      - Append our recommended ignore entries to .gitignore? (y/N)
#      - Which CLI/LLM should run INIT.md?
#      NEVER asks about team, domain, vision, personas, product purpose.
#      The agent infers those from the codebase via INIT.md.
#   3. Substitutes <PRODUCT_NAME>/<STACK> ONLY inside starter-managed paths.
#      (<TEAM>/<DOMAIN> placeholders are no longer populated by bootstrap;
#      INIT.md infers and writes them via the agent.)
#   4. NEVER overwrites/modifies pre-existing user files. Existing instruction
#      files are flagged in .starter-meta.json so INIT.md can read and improve
#      in place (essence preserved).
#   5. Hands off to the chosen CLI to execute INIT.md.
#
# Usage interactive:
#   ./bootstrap.sh
#
# Usage non-interactive (CI):
#   ./bootstrap.sh --yes --cli claude --append-gitignore yes

set -euo pipefail

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
NON_INTERACTIVE=0
CLI_PRESET=""
APPEND_GITIGNORE_PRESET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes)              NON_INTERACTIVE=1; shift ;;
    --cli)                 CLI_PRESET="$2"; shift 2 ;;
    --append-gitignore)    APPEND_GITIGNORE_PRESET="$2"; shift 2 ;;  # yes|no
    -h|--help)             sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# auto-detect stack
# ---------------------------------------------------------------------------
detect_stack_in() {
  local d="${1:-.}"
  if [[ -f "$d/angular.json" ]]; then echo "angular"; return; fi
  if [[ -f "$d/package.json" ]]; then
    if   grep -q '"next"'                "$d/package.json"; then echo "next-ts"
    elif grep -q '"@angular/core"'       "$d/package.json"; then echo "angular"
    elif grep -q '"react"'               "$d/package.json"; then echo "react-ts"
    elif grep -q '"vue"'                 "$d/package.json"; then echo "vue-ts"
    elif grep -q '"@nestjs/core"\|"nestjs"' "$d/package.json"; then echo "nestjs"
    elif grep -q '"express"'             "$d/package.json"; then echo "node-express"
    else echo "node-ts"
    fi
    return
  fi
  if compgen -G "$d/*.csproj" >/dev/null 2>&1 || compgen -G "$d/*.sln" >/dev/null 2>&1; then echo "dotnet"; return; fi
  if [[ -f "$d/pyproject.toml" || -f "$d/requirements.txt" ]]; then
    if   grep -qi 'django'  "$d/pyproject.toml" "$d/requirements.txt" 2>/dev/null; then echo "python-django"
    elif grep -qi 'fastapi' "$d/pyproject.toml" "$d/requirements.txt" 2>/dev/null; then echo "python-fastapi"
    elif grep -qi 'flask'   "$d/pyproject.toml" "$d/requirements.txt" 2>/dev/null; then echo "python-flask"
    else echo "python"
    fi
    return
  fi
  [[ -f "$d/go.mod" ]]        && { echo "go"; return; }
  [[ -f "$d/Cargo.toml" ]]    && { echo "rust"; return; }
  [[ -f "$d/pubspec.yaml" ]]  && { echo "flutter"; return; }
  if [[ -f "$d/composer.json" ]]; then
    if grep -q 'laravel/framework' "$d/composer.json"; then echo "laravel"; else echo "php"; fi
    return
  fi
  [[ -f "$d/Gemfile" ]]          && { echo "ruby"; return; }
  [[ -f "$d/mix.exs" ]]          && { echo "elixir"; return; }
  [[ -f "$d/build.gradle.kts" ]] && { echo "kotlin-gradle"; return; }
  [[ -f "$d/build.gradle" ]]     && { echo "java-gradle"; return; }
  [[ -f "$d/pom.xml" ]]          && { echo "java-maven"; return; }
  echo "unknown"
}

# ---------------------------------------------------------------------------
# auto-detect product name from manifest files (no human input)
# ---------------------------------------------------------------------------
read_json_field() {
  local file="$1" field="$2"
  [[ -f "$file" ]] || return 1
  grep -m1 "\"$field\"[[:space:]]*:" "$file" 2>/dev/null \
    | sed -E "s/.*\"$field\"[[:space:]]*:[[:space:]]*\"([^\"]+)\".*/\1/" \
    | head -1
}

read_toml_field() {
  local file="$1" field="$2"
  [[ -f "$file" ]] || return 1
  grep -m1 "^[[:space:]]*$field[[:space:]]*=" "$file" 2>/dev/null \
    | sed -E "s/^[[:space:]]*$field[[:space:]]*=[[:space:]]*\"([^\"]+)\".*/\1/" \
    | head -1
}

read_yaml_field() {
  local file="$1" field="$2"
  [[ -f "$file" ]] || return 1
  grep -m1 "^[[:space:]]*$field[[:space:]]*:" "$file" 2>/dev/null \
    | sed -E "s/^[[:space:]]*$field[[:space:]]*:[[:space:]]*\"?([^\"#[:space:]]+)\"?.*/\1/" \
    | head -1
}

detect_product_name_in() {
  local d="${1:-.}" name=""

  name="$(read_json_field "$d/package.json" "name" || true)"
  [[ -n "$name" ]] && { echo "$name"; return; }

  if [[ -f "$d/angular.json" ]]; then
    name="$(grep -A1 '"projects"[[:space:]]*:' "$d/angular.json" 2>/dev/null \
      | grep -oE '"[A-Za-z0-9_-]+"' | head -1 | tr -d '"')"
    [[ -n "$name" && "$name" != "projects" ]] && { echo "$name"; return; }
  fi

  if compgen -G "$d/*.csproj" >/dev/null 2>&1; then
    name="$(basename "$(ls -1 "$d"/*.csproj | head -1)" .csproj)"
    [[ -n "$name" ]] && { echo "$name"; return; }
  fi

  name="$(read_toml_field "$d/pyproject.toml" "name" || true)"
  [[ -n "$name" ]] && { echo "$name"; return; }

  name="$(read_toml_field "$d/Cargo.toml" "name" || true)"
  [[ -n "$name" ]] && { echo "$name"; return; }

  name="$(read_yaml_field "$d/pubspec.yaml" "name" || true)"
  [[ -n "$name" ]] && { echo "$name"; return; }

  name="$(read_json_field "$d/composer.json" "name" || true)"
  [[ -n "$name" ]] && { echo "${name##*/}"; return; }

  if [[ -f "$d/go.mod" ]]; then
    name="$(grep -m1 '^module ' "$d/go.mod" | awk '{print $2}' | awk -F/ '{print $NF}')"
    [[ -n "$name" ]] && { echo "$name"; return; }
  fi

  if [[ "$d" == "." ]]; then
    basename "$PWD"
  else
    basename "$d"
  fi
}

# ---------------------------------------------------------------------------
# project mode (projects/ folder convention)
# ---------------------------------------------------------------------------
detect_project_mode() {
  if [[ ! -d projects ]]; then
    echo "root"; return
  fi
  if find projects -mindepth 1 -maxdepth 1 -type d ! -name ".*" | grep -q .; then
    echo "monorepo"
  else
    echo "root"
  fi
}

PROJECT_MODE="$(detect_project_mode)"

PROJECTS_JSON="[]"
if [[ "$PROJECT_MODE" == "monorepo" ]]; then
  entries=()
  while IFS= read -r dir; do
    pn="$(detect_product_name_in "$dir")"
    st="$(detect_stack_in "$dir")"
    entries+=("{\"name\":\"$pn\",\"path\":\"$dir\",\"stack\":\"$st\"}")
  done < <(find projects -mindepth 1 -maxdepth 1 -type d ! -name ".*" | sort)
  PROJECTS_JSON="[$(IFS=,; echo "${entries[*]}")]"
  PRODUCT_NAME="$(basename "$PWD")"
  STACK="monorepo"
else
  PRODUCT_NAME="$(detect_product_name_in .)"
  STACK="$(detect_stack_in .)"
fi

echo "=========================================="
echo "  Agentic Starter - Bootstrap"
echo "=========================================="
echo ""
echo "Auto-detected (agent will infer team/domain/personas/vision from code):"
echo "  PROJECT_MODE: $PROJECT_MODE"
echo "  PRODUCT_NAME: $PRODUCT_NAME"
echo "  STACK:        $STACK"
if [[ "$PROJECT_MODE" == "monorepo" ]]; then
  echo "  PROJECTS:     $PROJECTS_JSON"
fi
echo ""

# ---------------------------------------------------------------------------
# detect existing instruction files (DO NOT overwrite — flag for INIT.md)
# ---------------------------------------------------------------------------
EXISTING_INSTRUCTION_FILES=()
for f in AGENTS.md CLAUDE.md INIT.md .github/copilot-instructions.md; do
  if [[ -f "$f" ]]; then
    if ! grep -q 'Agentic Starter\|<PRODUCT_NAME>\|<STACK>' "$f" 2>/dev/null; then
      EXISTING_INSTRUCTION_FILES+=("$f")
    fi
  fi
done

if (( ${#EXISTING_INSTRUCTION_FILES[@]} > 0 )); then
  echo "Detected pre-existing instruction files (will be preserved):"
  for f in "${EXISTING_INSTRUCTION_FILES[@]}"; do
    echo "  - $f"
  done
  echo "  -> INIT.md will READ them and IMPROVE in place (essence preserved)."
  echo ""
fi

# ---------------------------------------------------------------------------
# substitute <PRODUCT_NAME>/<STACK> ONLY in starter-managed paths
# <TEAM>/<DOMAIN> are left as-is for the agent to fill in.
# ---------------------------------------------------------------------------
STARTER_DIRS=(.specs .agents .skills .claude .codex)
STARTER_GITHUB_PATTERNS=(
  ".github/copilot-instructions.md"
  ".github/copilot"
  ".github/PULL_REQUEST_TEMPLATE.md"
  ".github/ISSUE_TEMPLATE"
  ".github/workflows/ci.yml"
  ".github/workflows/dod.yml"
)
STARTER_ROOT_FILES=(
  AGENTS.md CLAUDE.md INIT.md _BOOTSTRAP.md
  README.md README.pt-BR.md
  playwright.config.ts
)

TOUCHED=0

substitute_in_file() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  grep -Iq . "$f" 2>/dev/null || return 0
  grep -q '<PRODUCT_NAME>\|<STACK>' "$f" 2>/dev/null || return 0
  if sed --version >/dev/null 2>&1; then
    sed -i \
      -e "s|<PRODUCT_NAME>|$PRODUCT_NAME|g" \
      -e "s|<STACK>|$STACK|g" \
      "$f"
  else
    sed -i '' \
      -e "s|<PRODUCT_NAME>|$PRODUCT_NAME|g" \
      -e "s|<STACK>|$STACK|g" \
      "$f"
  fi
  TOUCHED=$((TOUCHED+1))
}

echo "Substituting <PRODUCT_NAME> / <STACK> inside starter-managed paths..."

for dir in "${STARTER_DIRS[@]}"; do
  if [[ -d "$dir" ]]; then
    while IFS= read -r f; do
      substitute_in_file "$f"
    done < <(find "$dir" -type f \( -name "*.md" -o -name "*.json" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" -o -name "*.ts" \))
  fi
done

for p in "${STARTER_GITHUB_PATTERNS[@]}"; do
  if [[ -d "$p" ]]; then
    while IFS= read -r f; do
      substitute_in_file "$f"
    done < <(find "$p" -type f)
  elif [[ -f "$p" ]]; then
    substitute_in_file "$p"
  fi
done

for f in "${STARTER_ROOT_FILES[@]}"; do
  substitute_in_file "$f"
done

echo "-> $TOUCHED files updated (only starter-managed paths)."
echo ""

# ---------------------------------------------------------------------------
# .gitignore — NEVER overwrite. Append (or create) on opt-in only.
# ---------------------------------------------------------------------------
RECOMMENDED_IGNORES='# === Agentic Starter (auto-managed) — do not remove this header ===
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
'

handle_gitignore() {
  local choice="${APPEND_GITIGNORE_PRESET:-}"
  if [[ -z "$choice" && "$NON_INTERACTIVE" == "0" ]]; then
    echo "=========================================="
    echo "  .gitignore"
    echo "=========================================="
    if [[ -f .gitignore ]]; then
      echo "An existing .gitignore was found."
      echo "I can APPEND recommended entries (your existing content is NEVER modified)."
    else
      echo "No .gitignore found. I can CREATE one with recommended entries."
    fi
    read -r -p "Proceed? [y/N]: " ans
    ans="${ans:-n}"
    case "${ans:0:1}" in
      y|Y|s|S) choice="yes" ;;
      *)       choice="no"  ;;
    esac
    echo ""
  fi
  choice="${choice:-no}"

  if [[ "$choice" != "yes" ]]; then
    echo "-> .gitignore left untouched."
    return
  fi

  if [[ -f .gitignore ]]; then
    if grep -q "Agentic Starter (auto-managed)" .gitignore 2>/dev/null; then
      echo "-> Recommended entries already present in .gitignore. Nothing to do."
    else
      printf '\n%s\n' "$RECOMMENDED_IGNORES" >> .gitignore
      echo "-> Recommended entries APPENDED to .gitignore (original content preserved)."
    fi
  else
    printf '%s\n' "$RECOMMENDED_IGNORES" > .gitignore
    echo "-> .gitignore CREATED."
  fi
}

handle_gitignore
echo ""

# ---------------------------------------------------------------------------
# .starter-meta.json (machine-readable handoff for INIT.md)
# ---------------------------------------------------------------------------
existing_files_json="[]"
if (( ${#EXISTING_INSTRUCTION_FILES[@]} > 0 )); then
  existing_files_json="["
  for f in "${EXISTING_INSTRUCTION_FILES[@]}"; do
    existing_files_json+="\"$f\","
  done
  existing_files_json="${existing_files_json%,}]"
fi

cat > .starter-meta.json <<EOF
{
  "product_name": "$PRODUCT_NAME",
  "stack": "$STACK",
  "project_mode": "$PROJECT_MODE",
  "projects": $PROJECTS_JSON,
  "bootstrapped_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "starter_version": "0.1.11",
  "existing_instruction_files": $existing_files_json,
  "init_must_ask": [],
  "init_must_infer": ["team", "domain", "vision_oneliner", "personas_beyond_dev"],
  "default_persona": "developer",
  "init_must_merge": $existing_files_json,
  "read_only_globs": ["**/*.razor", "**/*.cs", "**/*.csproj", "**/*.sln", "package.json", "pnpm-lock.yaml", "yarn.lock", "package-lock.json", "**/*.py", "**/*.go", "**/*.rs", "**/*.java", "**/*.kt", "**/*.dart", "**/*.php", "**/*.rb"]
}
EOF
echo "-> .starter-meta.json saved."
echo ""

chmod +x .claude/hooks/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# choose CLI / LLM
# ---------------------------------------------------------------------------
INIT_PROMPT='Read INIT.md and execute it. Do NOT modify any user source files (.razor, .cs, .ts, .py, .go, .rs, package.json, etc). Only write inside .specs/, .agents/, .skills/, .claude/, .codex/, .github/copilot*, .github/workflows/dod.yml plus root AGENTS.md/CLAUDE.md/INIT.md/README*.md. If AGENTS.md/CLAUDE.md/copilot-instructions.md already existed before bootstrap (see .starter-meta.json), READ them and IMPROVE in place — preserve their essence. DO NOT ask the human about team, domain, vision, personas, or product purpose: infer ALL of them by reading the codebase (README, package.json/angular.json/*.csproj/pyproject.toml/etc, entry points, routes, tests, env.example). Default persona is "developer"; additional personas must be derived from code (auth roles, route guards, UI flows, customer-facing copy). Honor projects/ convention: if .starter-meta.json.project_mode == "monorepo", iterate over .starter-meta.json.projects[] and produce per-project .specs/. Use parallel multi-agents.'

declare -a CLI_OPTS=(
  "claude|Claude Code|claude"
  "codex|Codex CLI (OpenAI)|codex"
  "cursor|Cursor (cursor-agent, Cursor 3.0+)|cursor-agent"
  "vscode|VS Code Agent Mode (paste into Chat)|code"
  "windsurf|Windsurf / Cascade (Codeium)|windsurf"
  "kiro|Kiro (AWS, paste into Chat)|kiro"
  "copilot|GitHub Copilot CLI (chat — no agent loop)|gh"
  "deepseek|Deepseek (via aider --model deepseek/deepseek-coder)|aider"
  "kimi|Kimi K2.6 (via aider --model openrouter/moonshotai/kimi-k2)|aider"
  "minimax|MiniMax M2.7 (via aider --model openrouter/minimax/minimax-text-01)|aider"
  "glm|GLM 5.1 (via aider --model openrouter/z-ai/glm-4.5)|aider"
  "hermes|Hermes Agent (Nous Research)|hermes"
  "openclaw|OpenClaw|openclaw"
  "aider|Aider (pick model interactively)|aider"
  "other|Other / manual (copy prompt to clipboard)|"
  "skip|Skip — I will run INIT.md later|"
)

choose_cli() {
  if [[ -n "$CLI_PRESET" ]]; then echo "$CLI_PRESET"; return; fi
  if [[ "$NON_INTERACTIVE" == "1" ]]; then echo "skip"; return; fi

  {
    echo "=========================================="
    echo "  Choose the CLI/LLM to run INIT.md"
    echo "=========================================="
    echo ""
    local i=1
    for opt in "${CLI_OPTS[@]}"; do
      IFS='|' read -r key label cmd <<< "$opt"
      local mark=""
      if [[ -n "$cmd" ]] && command -v "$cmd" >/dev/null 2>&1; then
        mark="  [installed]"
      fi
      printf "  [%2d] %s%s\n" "$i" "$label" "$mark"
      i=$((i+1))
    done
    echo ""
  } >&2

  local default_idx="${#CLI_OPTS[@]}"   # last option = "skip"
  read -r -p "Number [$default_idx]: " idx
  idx="${idx:-$default_idx}"
  if ! [[ "$idx" =~ ^[0-9]+$ ]] || (( idx < 1 || idx > ${#CLI_OPTS[@]} )); then
    idx="$default_idx"
  fi
  IFS='|' read -r key _ _ <<< "${CLI_OPTS[$((idx-1))]}"
  echo "$key"
}

CLI_CHOICE="$(choose_cli)"

# ---------------------------------------------------------------------------
# clipboard helper (best-effort)
# ---------------------------------------------------------------------------
copy_to_clipboard() {
  local text="$1"
  if   command -v pbcopy   >/dev/null 2>&1; then printf '%s' "$text" | pbcopy
  elif command -v xclip    >/dev/null 2>&1; then printf '%s' "$text" | xclip -selection clipboard
  elif command -v wl-copy  >/dev/null 2>&1; then printf '%s' "$text" | wl-copy
  elif command -v clip.exe >/dev/null 2>&1; then printf '%s' "$text" | clip.exe
  else return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# handoff
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Handing off to: $CLI_CHOICE"
echo "=========================================="
echo ""

case "$CLI_CHOICE" in
  claude)
    command -v claude >/dev/null 2>&1 || { echo "Claude Code not installed: https://docs.claude.com/claude-code"; exit 1; }
    exec claude "$INIT_PROMPT"
    ;;
  codex)
    command -v codex >/dev/null 2>&1 || { echo "Codex CLI not installed: https://github.com/openai/codex"; exit 1; }
    exec codex exec "$INIT_PROMPT"
    ;;
  copilot)
    command -v gh >/dev/null 2>&1 || { echo "gh CLI not installed: https://cli.github.com"; exit 1; }
    copy_to_clipboard "$INIT_PROMPT" && echo "Prompt copied to clipboard." || echo "(clipboard unavailable — copy manually below)"
    echo ""
    echo "GitHub Copilot CLI does not run an autonomous agent loop."
    echo "Open Copilot Chat (VS Code / IDE) and paste the prompt:"
    echo ""
    echo "  $INIT_PROMPT"
    echo ""
    ;;
  cursor)
    command -v cursor-agent >/dev/null 2>&1 || { echo "Cursor Agent CLI not installed (Cursor 3.0+)."; exit 1; }
    exec cursor-agent "$INIT_PROMPT"
    ;;
  vscode)
    copy_to_clipboard "$INIT_PROMPT" && echo "Prompt copied to clipboard." || echo "(clipboard unavailable — copy manually below)"
    echo ""
    echo "VS Code Agent Mode runs in-IDE (no autonomous CLI loop)."
    echo "1) Open this folder in VS Code."
    echo "2) Open Chat (Cmd+Ctrl+I / Ctrl+Alt+I)."
    echo "3) Switch mode to 'Agent'."
    echo "4) Paste the prompt:"
    echo ""
    echo "  $INIT_PROMPT"
    echo ""
    if command -v code >/dev/null 2>&1; then
      code . >/dev/null 2>&1 || true
    fi
    ;;
  windsurf)
    copy_to_clipboard "$INIT_PROMPT" && echo "Prompt copied to clipboard." || echo "(clipboard unavailable — copy manually below)"
    echo ""
    echo "Windsurf Cascade runs in-IDE."
    echo "1) Open this folder in Windsurf."
    echo "2) Open Cascade panel."
    echo "3) Switch to Write mode."
    echo "4) Paste the prompt."
    echo ""
    if command -v windsurf >/dev/null 2>&1; then
      windsurf . >/dev/null 2>&1 || true
    fi
    ;;
  kiro)
    copy_to_clipboard "$INIT_PROMPT" && echo "Prompt copied to clipboard." || echo "(clipboard unavailable — copy manually below)"
    echo ""
    echo "Kiro runs in-IDE."
    echo "1) Open this folder in Kiro."
    echo "2) Open Chat panel."
    echo "3) Paste the prompt and run in Agent mode."
    echo ""
    if command -v kiro >/dev/null 2>&1; then
      kiro . >/dev/null 2>&1 || true
    fi
    ;;
  deepseek)
    command -v aider >/dev/null 2>&1 || { echo "aider not installed: pipx install aider-chat"; exit 1; }
    exec aider --model deepseek/deepseek-coder --message "$INIT_PROMPT"
    ;;
  kimi)
    command -v aider >/dev/null 2>&1 || { echo "aider not installed: pipx install aider-chat"; exit 1; }
    exec aider --model openrouter/moonshotai/kimi-k2 --message "$INIT_PROMPT"
    ;;
  minimax)
    command -v aider >/dev/null 2>&1 || { echo "aider not installed: pipx install aider-chat"; exit 1; }
    exec aider --model openrouter/minimax/minimax-text-01 --message "$INIT_PROMPT"
    ;;
  glm)
    command -v aider >/dev/null 2>&1 || { echo "aider not installed: pipx install aider-chat"; exit 1; }
    exec aider --model openrouter/z-ai/glm-4.5 --message "$INIT_PROMPT"
    ;;
  hermes)
    command -v hermes >/dev/null 2>&1 || { echo "Hermes Agent not installed: https://github.com/NousResearch/hermes-agent"; exit 1; }
    copy_to_clipboard "$INIT_PROMPT" && echo "(prompt copied to clipboard as fallback)"
    exec hermes "$INIT_PROMPT"
    ;;
  openclaw)
    command -v openclaw >/dev/null 2>&1 || { echo "OpenClaw not installed: npm install -g openclaw@latest"; exit 1; }
    copy_to_clipboard "$INIT_PROMPT" && echo "(prompt copied to clipboard as fallback)"
    exec openclaw "$INIT_PROMPT"
    ;;
  aider)
    command -v aider >/dev/null 2>&1 || { echo "aider not installed: pipx install aider-chat"; exit 1; }
    exec aider --message "$INIT_PROMPT"
    ;;
  other)
    if copy_to_clipboard "$INIT_PROMPT"; then
      echo "Prompt copied to clipboard. Paste it into your CLI/agent of choice."
    else
      echo "(clipboard unavailable — copy the prompt below manually)"
    fi
    echo ""
    echo "Prompt:"
    echo "  $INIT_PROMPT"
    echo ""
    ;;
  skip|*)
    cat <<EOF
Skipped CLI handoff. To run INIT.md later, open your agent and paste:

  $INIT_PROMPT

Recommended next steps:
  1) Open an agent in this folder.
  2) Paste the prompt above.
  3) Review .specs/product/VISION.md, DOMAIN.md, architecture/DESIGN.md.
  4) git add -A && git commit -m "chore: bootstrap agentic starter"

Docs: https://github.com/wesleysimplicio/agentic-starter
EOF
    ;;
esac
