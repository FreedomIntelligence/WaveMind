#!/bin/bash

# preprocess_wavemind_dataset.sh
# Automated script to process all WaveMind EEG datasets sequentially
# Usage: ./preprocess_wavemind_dataset.sh [dataset_name] [options]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="${SCRIPT_DIR}/preprocess_errors.log"

# Dataset processing order (can be customized)
DATASETS=(
    "SEED"
    "Siena_Scalp_EEG_Database" 
    "THING-EEG"
    "TUAB"
    "TUEV"
    "ImageNetEEG"
    "CHB-MIT"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_message() {
    local message="$1"
    local level="${2:-INFO}"
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $level - $message" | tee -a "$LOG_FILE"
}

log_error() {
    log_message "$1" "ERROR"
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >> "$ERROR_LOG"
}

log_success() {
    log_message "$1" "SUCCESS"
}

log_warning() {
    log_message "$1" "WARNING"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python dependencies
check_python_dependencies() {
    log_message "Checking Python dependencies..."
    
    if ! command_exists python; then
        log_error "Python is not installed or not in PATH"
        return 1
    fi
    
    # Check for common packages used in process scripts
    local packages=("numpy" "mne" "torcheeg" "tqdm" "h5py" "PIL" "transformers")
    for pkg in "${packages[@]}"; do
        if ! python -c "import $pkg" 2>/dev/null; then
            log_warning "Package $pkg may not be installed"
        fi
    done
    
    log_success "Python dependency check completed"
}

# Check if environment variable is set
check_environment() {
    if [[ -z "${WaveMind_ROOT_PATH_:-}" ]]; then
        log_warning "WaveMind_ROOT_PATH_ environment variable is not set"
        log_warning "Some datasets may require this variable to be set"
    fi
}

# Process individual dataset
process_dataset() {
    local dataset="$1"
    local dataset_path="${SCRIPT_DIR}/${dataset}"
    
    log_message "Processing dataset: $dataset"
    
    if [[ ! -d "$dataset_path" ]]; then
        log_error "Dataset directory not found: $dataset_path"
        return 1
    fi
    
    cd "$dataset_path" || {
        log_error "Failed to change directory to: $dataset_path"
        return 1
    }
    
    case "$dataset" in
        "CHB-MIT")
            process_chb_mit
            ;;
        "ImageNetEEG")
            process_imagenet_eeg
            ;;
        "SEED")
            process_seed
            ;;
        "Siena_Scalp_EEG_Database")
            process_siena
            ;;
        "THING-EEG")
            process_thing_eeg
            ;;
        "TUAB")
            process_tuab
            ;;
        "TUEV")
            process_tuev
            ;;
        *)
            log_error "Unknown dataset: $dataset"
            return 1
            ;;
    esac
    
    cd "$SCRIPT_DIR" || return 1
}

# Dataset-specific processing functions
process_chb_mit() {
    log_message "Processing CHB-MIT dataset (3 stages)"
    
    local processes=("process1.py" "process2.py" "process3.py")
    
    for script in "${processes[@]}"; do
        if [[ ! -f "$script" ]]; then
            log_error "Script not found: $script"
            return 1
        fi
        
        log_message "Running $script..."
        if python "$script"; then
            log_success "Completed: $script"
        else
            log_error "Failed: $script (exit code: $?)"
            return 1
        fi
    done
}

process_imagenet_eeg() {
    log_message "Processing ImageNetEEG dataset"
    
    # Check if download is needed
    if [[ -f "download.py" ]]; then
        log_message "Checking if download is needed..."
        if python -c "
import os
image_files = [f for f in os.listdir('Image') if f.endswith('.JPEG')] if os.path.exists('Image') else []
print(f'Found {len(image_files)} images')
exit(1 if len(image_files) == 0 else 0)
" 2>/dev/null; then
            log_message "Images already downloaded, skipping download"
        else
            log_message "Running download.py..."
            python download.py || {
                log_warning "Download may have failed or images may already exist"
            }
        fi
    fi
    
    # Run main processing
    if [[ -f "process.py" ]]; then
        log_message "Running process.py..."
        python process.py
    else
        log_error "process.py not found"
        return 1
    fi
}

process_seed() {
    log_message "Processing SEED dataset"
    
    if [[ -f "process.py" ]]; then
        python process.py
    else
        log_error "process.py not found"
        return 1
    fi
}

process_siena() {
    log_message "Processing Siena Scalp EEG Database"
    
    if [[ -f "process.py" ]]; then
        python process.py
    else
        log_error "process.py not found"
        return 1
    fi
}

process_thing_eeg() {
    log_message "Processing THING-EEG dataset"
    
    if [[ -f "process.py" ]]; then
        python process.py
    else
        log_error "process.py not found"
        return 1
    fi
}

process_tuab() {
    log_message "Processing TUAB dataset"
    
    # Check for multiple processing scripts
    if [[ -f "process.py" ]]; then
        # Check if it supports stage parameter
        if grep -q "stage" process.py; then
            log_message "Running TUAB process in stages..."
            for stage in 1 2 3; do
                log_message "Running stage $stage..."
                python process.py --stage "$stage" || {
                    log_error "Failed stage $stage"
                    return 1
                }
            done
        else
            python process.py
        fi
    else
        log_error "process.py not found"
        return 1
    fi
}

process_tuev() {
    log_message "Processing TUEV dataset"
    
    # Check if download is needed
    if [[ -f "download.py" ]]; then
        log_message "Checking if download is needed..."
        if [[ ! -d "edf" ]]; then
            log_message "Running download.py..."
            python download.py
        else
            log_message "EDF files already downloaded"
        fi
    fi
    
    if [[ -f "process.py" ]]; then
        python process.py
    else
        log_error "process.py not found"
        return 1
    fi
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS] [DATASET_NAME]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -l, --list          List available datasets"
    echo "  -s, --skip-check    Skip dependency checks"
    echo "  DATASET_NAME        Process only specific dataset"
    echo ""
    echo "Available datasets:"
    for dataset in "${DATASETS[@]}"; do
        echo "  - $dataset"
    done
}

# Main execution function
main() {
    local specific_dataset=""
    local skip_check=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_usage
                exit 0
                ;;
            -l|--list)
                echo "Available datasets:"
                for dataset in "${DATASETS[@]}"; do
                    echo "  - $dataset"
                done
                exit 0
                ;;
            -s|--skip-check)
                skip_check=true
                shift
                ;;
            *)
                if [[ -z "$specific_dataset" ]]; then
                    specific_dataset="$1"
                else
                    echo "Error: Too many arguments"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Initialize logging
    echo "WaveMind Dataset Preprocessing Log" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "==================================" >> "$LOG_FILE"
    
    log_message "Starting WaveMind dataset preprocessing"
    log_message "Log file: $LOG_FILE"
    log_message "Error log: $ERROR_LOG"
    
    # Check dependencies unless skipped
    if [[ "$skip_check" == false ]]; then
        check_python_dependencies
        check_environment
    fi
    
    # Process specific dataset or all datasets
    if [[ -n "$specific_dataset" ]]; then
        if [[ " ${DATASETS[*]} " == *" $specific_dataset "* ]]; then
            log_message "Processing specific dataset: $specific_dataset"
            process_dataset "$specific_dataset"
        else
            log_error "Unknown dataset: $specific_dataset"
            show_usage
            exit 1
        fi
    else
        log_message "Processing all datasets in order"
        for dataset in "${DATASETS[@]}"; do
            if process_dataset "$dataset"; then
                log_success "Completed processing: $dataset"
            else
                log_error "Failed to process: $dataset"
                log_message "Continuing with next dataset..."
            fi
        done
    fi
    
    log_success "All dataset processing completed"
    
    # Run final dataset creation for RAG module
    log_message "Running final dataset creation..."
    if [[ -f "create_dataset_pkl_npy.py" ]]; then
        if python create_dataset_pkl_npy.py; then
            log_success "Final dataset creation completed"
        else
            log_error "Final dataset creation failed"
        fi
    else
        log_warning "create_dataset_pkl_npy.py not found, skipping final step"
    fi
    
    echo -e "${GREEN}Preprocessing completed successfully!${NC}"
    echo -e "Check logs: $LOG_FILE"
    
    if [[ -s "$ERROR_LOG" ]]; then
        echo -e "${YELLOW}Warnings and errors were logged: $ERROR_LOG${NC}"
    fi
}

# Handle script termination
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script terminated with error code: $exit_code"
        echo -e "${RED}Script failed! Check logs: $LOG_FILE${NC}"
    fi
    exit $exit_code
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Run main function with all arguments
main "$@"