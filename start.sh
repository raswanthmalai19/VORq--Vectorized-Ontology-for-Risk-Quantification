#!/bin/bash
# ─── VORQ Launch Script ─────────────────────────────────────────────────────
# Starts both FastAPI backend and Streamlit frontend.

set -e

# M1 Mac optimization
CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
THREADS=$((CPU_CORES / 2))
if [ "$THREADS" -lt 2 ]; then THREADS=2; fi
if [ "$THREADS" -gt 8 ]; then THREADS=8; fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export VORQ_ENABLE_GPU=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🌍 VORQ — Global Risk Intelligence"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Kill existing processes
pkill -f "uvicorn vorq.api.main" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
sleep 1

# Start API backend
echo "🚀 Starting FastAPI backend on port 8000..."
python3 -m uvicorn vorq.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
echo "   API PID: $API_PID"

# Wait for API to be ready
echo "⏳ Waiting for API..."
for i in $(seq 1 15); do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit UI
echo "🎨 Starting Streamlit dashboard on port 8501..."
python3 -m streamlit run vorq/ui/app.py --server.port 8501 --server.headless true &
UI_PID=$!
echo "   UI PID: $UI_PID"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Dashboard: http://localhost:8501"
echo "📡 API Docs:  http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services."

# Handle shutdown
trap "kill $API_PID $UI_PID 2>/dev/null; echo ''; echo '👋 VORQ stopped.'" EXIT

wait
