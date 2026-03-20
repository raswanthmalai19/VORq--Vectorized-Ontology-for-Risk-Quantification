#!/bin/bash
# ─── Download trained models from Google Drive to local Mac ───
# Run this after training in Colab.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_MODEL_DIR="$PROJECT_DIR/vorq/models/event_classifier"

echo "🌍 VORQ — Model Download"
echo "━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Option 1: If Google Drive Desktop is installed
GDRIVE_PATH="$HOME/Google Drive/My Drive/vorq_models/event_classifier"
GDRIVE_PATH_ALT="$HOME/Library/CloudStorage/GoogleDrive-*/My Drive/vorq_models/event_classifier"

mkdir -p "$LOCAL_MODEL_DIR"

if [ -d "$GDRIVE_PATH" ]; then
    echo "📁 Found models in Google Drive Desktop..."
    cp -r "$GDRIVE_PATH"/* "$LOCAL_MODEL_DIR/"
    echo "✅ Models copied to: $LOCAL_MODEL_DIR"
elif compgen -G "$GDRIVE_PATH_ALT" > /dev/null 2>&1; then
    FOUND_PATH=$(compgen -G "$GDRIVE_PATH_ALT" | head -1)
    echo "📁 Found models in Google Drive ($FOUND_PATH)..."
    cp -r "$FOUND_PATH"/* "$LOCAL_MODEL_DIR/"
    echo "✅ Models copied to: $LOCAL_MODEL_DIR"
else
    echo "⚠️  Google Drive not found locally."
    echo ""
    echo "Please download manually:"
    echo "  1. Go to Google Drive → vorq_models/event_classifier/"
    echo "  2. Download all files"
    echo "  3. Place them in: $LOCAL_MODEL_DIR/"
    echo ""
    echo "Or use gdown:"
    echo "  pip install gdown"
    echo "  gdown --folder <google_drive_folder_url> -O $LOCAL_MODEL_DIR"
fi

echo ""
echo "Local model directory: $LOCAL_MODEL_DIR"
if [ -f "$LOCAL_MODEL_DIR/config.json" ]; then
    echo "✅ Model files present:"
    ls -la "$LOCAL_MODEL_DIR"
else
    echo "⚠️  No model files found yet. Train in Colab first."
fi
