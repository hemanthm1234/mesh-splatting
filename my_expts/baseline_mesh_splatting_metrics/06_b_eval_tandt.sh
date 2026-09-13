#!/bin/bash
set -e

EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
SCENE=${1:-"Truck"}
EVAL_SCENE="$SCENE"
if [ "$SCENE" = "Meeting_room" ] || [ "$SCENE" = "meeting_room" ]; then
    EVAL_SCENE="Meetingroom"
fi
DATASET_DIR="/data1/hemanth/datasets/tandt_eval_data/$EVAL_SCENE"
OUTPUT_DIR="$EXP_DIR/output/${SCENE}_baseline"
ITERATION=${2:-30000}
PLY_PATH="$OUTPUT_DIR/native_mesh_${ITERATION}.ply"
EVAL_OUT_DIR="$OUTPUT_DIR/tnt_eval_results"

if [ ! -f "$PLY_PATH" ]; then
    echo "[!] Error: Mesh file not found at $PLY_PATH"
    echo "    Please make sure 06_extract_native.sh ran successfully."
    exit 1
fi

echo "=== Running Tanks & Temples Evaluation on $PLY_PATH ==="

python "$EXP_DIR/eval_mesh_splatting.py" \
    --dataset-dir "$DATASET_DIR" \
    --ply-path "$PLY_PATH" \
    --out-dir "$EVAL_OUT_DIR"

echo "=== Tanks & Temples Evaluation Complete ==="
