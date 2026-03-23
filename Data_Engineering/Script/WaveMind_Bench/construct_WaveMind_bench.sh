#!/bin/bash

# Get the directory where this script is located as the working directory
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Change to the working directory
echo "Changing to working directory: $WORK_DIR"
cd "$WORK_DIR" || { echo "Failed to change to working directory $WORK_DIR"; exit 1; }

# Function to run a python script with error handling
run_python_script() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    local folder_name=$(dirname "$script_path" | xargs basename)
    
    if [ -f "$script_path" ]; then
        echo "Running [$folder_name] $script_name..."
        python "$script_path" || { echo "[$folder_name] Failed to run $script_name"; exit 1; }
        echo "[$folder_name] $script_name completed successfully"
    else
        echo "Error: [$folder_name] $script_name does not exist at $script_path"
        exit 1
    fi
    echo "----------------------------------------"
}

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# get WaveMind_ROOT_PATH_ for backward compatibility, otherwise use auto-detected
if [ -z "${WaveMind_ROOT_PATH_}" ]; then
    TARGET_PATH="${PROJECT_ROOT}/Data_Engineering/data/WaveMind_Bench"
else
    TARGET_PATH="${WaveMind_ROOT_PATH_}/Data_Engineering/data/WaveMind_Bench"
fi
echo "Target path to delete: $TARGET_PATH"
rm -rf "$TARGET_PATH"
echo "Deleted: $TARGET_PATH"



# Run all the benchmark scripts
run_python_script "ImageNetEEG/build_WaveMind_Bench.py"
run_python_script "SEED/build_WaveMind_Bench.py"
run_python_script "THING/build_WaveMind_Bench.py"
run_python_script "TUAB/build_WaveMind_Bench.py"
run_python_script "TUEV/build_WaveMind_Bench.py"

echo "All scripts executed successfully"