#!/bin/bash
set -e

# Directory for datasets
DATASET_DIR="/data1/hemanth/datasets"
TANDT_DIR="$DATASET_DIR/tandt"

mkdir -p "$DATASET_DIR"

echo "Downloading Tanks & Temples (Preprocessed with COLMAP by INRIA)..."
cd "$DATASET_DIR"

if [ ! -d "tandt/truck" ]; then
    if ! command -v aria2c >/dev/null 2>&1; then
        echo "aria2c not found. Attempting to install aria2 for maximum parallel download speed..."
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y aria2 || true
        fi
        if ! command -v aria2c >/dev/null 2>&1; then
            echo "Failed to install aria2 automatically. Please install it manually for max speed."
        fi
    fi

    if command -v aria2c >/dev/null 2>&1; then
        echo "Using aria2c for fast multi-connection (16x) download..."
        aria2c -x 16 -s 16 -c https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
    else
        echo "Falling back to wget..."
        wget -c https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
    fi

    echo "Unzipping tandt_db.zip..."
    unzip -q -o tandt_db.zip
else
    echo "INRIA tandt/truck directory already exists. Skipping download and unzip."
fi

echo "Tanks & Temples dataset is ready at $TANDT_DIR"
echo "Specifically, the truck scene is at $TANDT_DIR/truck"
