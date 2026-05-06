#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Biosecurity Sentinel — Startup Script
# Usage: chmod +x scripts/start.sh && ./scripts/start.sh
# ═══════════════════════════════════════════════════════════════

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║       Biosecurity Sentinel v1.0           ║"
echo "║  3D Molecular Dual-Use Threat Screening   ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Check Python
python3 -c "import fastapi" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt --quiet
}

# Create required directories
mkdir -p logs exports models/weights data

# Copy .env if not exists
[ -f .env ] || { cp .env.template .env; echo "📝 Created .env from template"; }

echo "🚀 Starting API on port 8001..."
echo "   Docs: http://localhost:8001/docs"
echo "   Demo: open notebooks/demo.ipynb"
echo ""

python3 -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --log-level info \
    --reload
