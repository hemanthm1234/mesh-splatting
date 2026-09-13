#!/bin/bash
set -e

# Ensure we are in the root of the mesh-splatting repository
cd /data1/hemanth/mesh-splatting

# Base directory for Tanks & Temples dataset
BASE_DIR="/data1/hemanth/datasets/tandt"
# Available scenes in dataset
ALL_AVAILABLE_SCENES=("Barn" "Caterpillar" "Church" "Ignatius" "Meeting_room" "Truck")

# Output directory for experiment artifacts
OUTPUT_DIR="/data1/hemanth/mesh-splatting/my_expts/vggt_omega_initialised/outputs"
mkdir -p "$OUTPUT_DIR"

# Print usage helper
usage() {
    echo "================================================================="
    echo "VGGT Omega Experiment Runner"
    echo "================================================================="
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --scene SCENE [SCENE2 ...]   Scene(s) to process (e.g. Barn, Caterpillar, Church, Ignatius, Meeting_room, Truck, or 'all')"
    echo "  -m, --mode MODE                  Experiment mode: 'full_pipeline', 'geometry_only', or 'both' (default: both)"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --scene Barn"
    echo "  $0 --scene Barn Caterpillar"
    echo "  $0 --scene all"
    echo "  $0 --scene Church --mode full_pipeline"
    echo "================================================================="
    exit 0
}

SELECTED_INPUTS=()
MODE="both"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--scene)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                SELECTED_INPUTS+=("$1")
                shift
            done
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            # Allow positional arguments for scene names
            SELECTED_INPUTS+=("$1")
            shift
            ;;
    esac
done

# Resolve scenes to process
TARGET_SCENES=()

if [ ${#SELECTED_INPUTS[@]} -eq 0 ]; then
    echo "ℹ️ No scene argument provided. Defaulting to all available scenes: ${ALL_AVAILABLE_SCENES[*]}"
    TARGET_SCENES=("${ALL_AVAILABLE_SCENES[@]}")
else
    for INPUT in "${SELECTED_INPUTS[@]}"; do
        LOWER_INPUT=$(echo "$INPUT" | tr '[:upper:]' '[:lower:]')
        if [ "$LOWER_INPUT" == "all" ]; then
            TARGET_SCENES=("${ALL_AVAILABLE_SCENES[@]}")
            break
        fi

        MATCH_FOUND=false
        for KNOWN in "${ALL_AVAILABLE_SCENES[@]}"; do
            LOWER_KNOWN=$(echo "$KNOWN" | tr '[:upper:]' '[:lower:]')
            if [ "$LOWER_INPUT" == "$LOWER_KNOWN" ]; then
                TARGET_SCENES+=("$KNOWN")
                MATCH_FOUND=true
                break
            fi
        done

        if [ "$MATCH_FOUND" = false ]; then
            echo "❌ Error: Unknown scene '$INPUT'."
            echo "Available valid scenes are: ${ALL_AVAILABLE_SCENES[*]} (or 'all')"
            exit 1
        fi
    done
fi

# Deduplicate target scenes
SCENES=()
for SCENE in "${TARGET_SCENES[@]}"; do
    if [[ ! " ${SCENES[*]} " =~ " ${SCENE} " ]]; then
        SCENES+=("$SCENE")
    fi
done

echo "================================================================="
echo "Starting VGGT Omega Experiments"
echo "Target Scene(s): ${SCENES[*]}"
echo "Selected Mode  : $MODE"
echo "================================================================="

for SCENE in "${SCENES[@]}"; do
    SCENE_PATH="$BASE_DIR/$SCENE"

    echo ""
    echo "================================================================="
    echo "Processing Scene: $SCENE"
    echo "================================================================="

    # Step 1: Check Precomputed Depth Scaling Parameters
    echo ">> [1/3] Checking Depth Scaling Parameters for $SCENE..."
    if [ ! -f "$SCENE_PATH/vggt_omega/depth_params.json" ]; then
        echo "❌ Error: depth_params.json not found for $SCENE!"
        echo "Please run ./precompute_all_depths.sh first for $SCENE."
        exit 1
    fi
    echo "✅ Depth params verified for $SCENE."

    # Step 2: Run VGGT Full Pipeline (Experiment 2)
    if [ "$MODE" == "full_pipeline" ] || [ "$MODE" == "both" ]; then
        RUN_MODE="full_pipeline"
        RUN_NAME="${SCENE}_vggt_${RUN_MODE}"
        MODEL_PATH="${OUTPUT_DIR}/${RUN_NAME}"

        echo ">> [2/3] Running Training: $RUN_NAME"
        python train.py \
            --source_path "$SCENE_PATH" \
            --model_path "$MODEL_PATH" \
            --wandb_name "$RUN_NAME" \
            --scene_name "${SCENE,,}" \
            --eval \
            --vggt_mode "$RUN_MODE"
    fi

    # Step 3: Deprecated VGGT Geometry Only
    if [ "$MODE" == "geometry_only" ]; then
        echo "ERROR: 'geometry_only' mode is DEPRECATED and UTTER NONSENSE. It must never be used!"
        echo "All VGGT models must use native VGGT cameras via --vggt_mode full_pipeline."
        exit 1
    fi

done

echo ""
echo "================================================================="
echo "All requested scenes processed successfully!"
echo "================================================================="
