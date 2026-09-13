#!/bin/bash
set -e

# Run all 6 successfully prepared Tanks & Temples scenes sequentially
# ./run_experiment.sh Barn
# ./run_experiment.sh Caterpillar
# ./run_experiment.sh Courthouse
./run_experiment.sh Ignatius
./run_experiment.sh Meeting_room
./run_experiment.sh Truck

echo "=== All 6 Baseline Experiments Compldeted Successfully! ==="
