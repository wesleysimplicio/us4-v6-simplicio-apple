# Hook pre-commit do Claude Code (PowerShell sibling de pre-commit.sh).
# Roda a suite de testes em modo silencioso e BLOQUEIA o commit se vermelho.
# Mensagens em pt-BR para feedback rapido. CI roda o set completo (lint + e2e);
# aqui o foco e evitar commit obviamente quebrado.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

Write-Host '[pre-commit] Rodando testes locais antes do commit...'

# Garante que existe package.json antes de tentar npm test.
if (-not (Test-Path -LiteralPath 'package.json' -PathType Leaf)) {
    Write-Host '[pre-commit] package.json nao encontrado. Pulando testes.'
    exit 0
}

# Roda npm test silenciosamente. Se falhar, bloqueia.
$npmCmd = if ($IsWindows) { 'npm.cmd' } else { 'npm' }

& $npmCmd test --silent
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host ''
    Write-Host '[pre-commit] FALHOU: testes vermelhos. Commit bloqueado.'
    Write-Host '[pre-commit] Corrija os testes antes de commitar.'
    Write-Host "[pre-commit] Para depurar: rode 'npm test' direto e leia o output."
    exit 1
}

Write-Host '[pre-commit] Testes verdes. Seguindo com o commit.'
exit 0
