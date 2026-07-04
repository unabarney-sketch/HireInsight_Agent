# ============================================================
# HireInsight_Agent - Interactive One-Click Deploy Script
# Safely commits and pushes to GitHub (supports 2FA / PAT)
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HireInsight_Agent Deploy Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# Step 1: Privacy Check
# ============================================================
Write-Host "[1/4] Running privacy checks..." -ForegroundColor Yellow

# 1a. Verify .gitignore exists
if (-not (Test-Path ".gitignore")) {
    Write-Host "  [FAIL] .gitignore not found! Create one first." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] .gitignore exists" -ForegroundColor Green

# 1b. Verify .gitignore covers .env
$gitignoreContent = Get-Content ".gitignore" -Raw
if ($gitignoreContent -match "^\.env" -or $gitignoreContent -match "`n\.env") {
    Write-Host "  [OK] .gitignore includes .env rule" -ForegroundColor Green
} else {
    Write-Host "  [WARN] .env rule not found in .gitignore!" -ForegroundColor Yellow
}

# 1c. Check if .env is tracked by Git (CRITICAL)
$trackedEnv = git ls-files .env 2>$null
if ($trackedEnv) {
    Write-Host "  [CRITICAL] .env is tracked by Git! Removing from tracking..." -ForegroundColor Red
    git rm --cached .env 2>$null
    Write-Host "  [OK] .env removed from Git tracking (local file kept)" -ForegroundColor Green
} else {
    Write-Host "  [OK] .env is NOT tracked by Git - safe" -ForegroundColor Green
}

# 1d. Check other sensitive files
$sensitiveFiles = @("config.json", "config.local.json", "secrets.json", "credentials.json")
foreach ($file in $sensitiveFiles) {
    if (Test-Path $file) {
        $trackedFile = git ls-files $file 2>$null
        if ($trackedFile) {
            Write-Host "  [WARN] $file is tracked, removing..." -ForegroundColor Yellow
            git rm --cached $file 2>$null
        }
    }
}

Write-Host ""
Write-Host "  [OK] Privacy check passed!" -ForegroundColor Green

# ============================================================
# Step 2: Stage Changes
# ============================================================
Write-Host ""
Write-Host "[2/4] Staging files..." -ForegroundColor Yellow

git add .
Write-Host "  [OK] git add . complete" -ForegroundColor Green

Write-Host ""
Write-Host "  Changes to be committed:" -ForegroundColor Cyan
git status --short
Write-Host ""

# ============================================================
# Step 3: Commit
# ============================================================
Write-Host "[3/4] Committing changes..." -ForegroundColor Yellow

$defaultMessage = "feat: automated deploy"
Write-Host ""
Write-Host "  Enter commit message (press Enter for default):" -ForegroundColor Cyan
Write-Host "  Default: $defaultMessage" -ForegroundColor DarkGray
$commitMessage = Read-Host "  Message"

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = $defaultMessage
}

git commit -m $commitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [INFO] Nothing to commit, skipping." -ForegroundColor Yellow
} else {
    Write-Host "  [OK] Commit successful" -ForegroundColor Green
}

# ============================================================
# Step 4: Push to GitHub (interactive PAT input)
# ============================================================
Write-Host ""
Write-Host "[4/4] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  GitHub 2FA is enabled - you need a Personal Access Token." -ForegroundColor Yellow
Write-Host "  If you don't have one, generate it at:" -ForegroundColor Cyan
Write-Host "     https://github.com/settings/tokens" -ForegroundColor DarkGray
Write-Host "  (Create a classic token, check 'repo' scope)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Paste your GitHub Personal Access Token:" -ForegroundColor Cyan
Write-Host "  (input is hidden, press Enter when done)" -ForegroundColor DarkGray
$secureToken = Read-Host "  Token" -AsSecureString

$token = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
)

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "  [ABORT] No token provided. Push cancelled." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  Pushing to GitHub..." -ForegroundColor Cyan

# Save original remote URL
$originalUrl = git remote get-url origin
$pushUrl = "https://unabarney-sketch:$token@github.com/unabarney-sketch/HireInsight_Agent.git"

# Push with token-embedded URL
$pushResult = git push -u $pushUrl main 2>&1
$pushExitCode = $LASTEXITCODE

# IMMEDIATELY restore clean URL (token never persists in git config)
git remote set-url origin $originalUrl 2>$null

if ($pushExitCode -eq 0) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  SUCCESS! Code pushed to GitHub." -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Repo: https://github.com/unabarney-sketch/HireInsight_Agent" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  PUSH FAILED!" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Error: $pushResult" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Troubleshooting:" -ForegroundColor Cyan
    Write-Host "  1. Invalid/expired token -> regenerate: https://github.com/settings/tokens" -ForegroundColor DarkGray
    Write-Host "  2. Token missing 'repo' scope -> check 'repo' when generating" -ForegroundColor DarkGray
    Write-Host "  3. Network issue -> check proxy/VPN settings" -ForegroundColor DarkGray
    Write-Host "  4. Remote repo not found -> verify repo name" -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}
