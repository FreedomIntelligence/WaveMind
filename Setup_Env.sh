#!/bin/bash

# Check if WaveMind_ROOT_PATH_ parameter is provided
if [ -z "$1" ]; then
    echo "Add project root_path to the end of this bash. Usage: $0 <WaveMind_ROOT_PATH_>"
    exit 1
fi

# Get the path specified by the user
WaveMind_ROOT_PATH_="$1"

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Check if the required project directory exists
if [ ! -d "$SCRIPT_DIR/EEGLLM" ]; then
    echo "Project build fail, repull" >&2
    exit 1
fi

# Change to the script's directory
cd "$SCRIPT_DIR"

# Uninstall existing packages

pip uninstall WaveMind -y
pip uninstall llava -y




# Save the current working directory
original_path=$(pwd)
pip install uv
# Install the current project
uv pip install -e . --quiet


# Return to original directory
cd "$original_path" || exit

# Change to LLaVA directory and install
cd EEGLLM/LLaVA || { echo "Error: Cannot enter EEGLLM/LLaVA directory"; exit 1; }
uv pip install -e .  --quiet

# Return to original directory
cd "$original_path" || exit

# Add WaveMind_ROOT_PATH_ to ~/.bashrc if not already present
if ! grep -q "export WaveMind_ROOT_PATH_=" ~/.bashrc; then
    echo "export WaveMind_ROOT_PATH_=$WaveMind_ROOT_PATH_" >> ~/.bashrc
    echo "✅ WaveMind_ROOT_PATH_ added to ~/.bashrc"
else
    echo "ℹ️ WaveMind_ROOT_PATH_ already exists in ~/.bashrc"
fi

# Reload the shell configuration
source ~/.bashrc

# Completion message
echo "All installation operations completed successfully"