#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/../../engine_integration" && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

# -----------------------------------------------------------------------------
# Defaults & Globals
# -----------------------------------------------------------------------------
DEFAULT_MODEL="meta-llama/Llama-3.2-1B"
DEFAULT_PORT_VLLM=12346
DEFAULT_PORT_SGL=30000

NUM_PROMPTS=1000
REQUEST_RATE=10

# CLI variables
engine=""        # positional
port=""
model=""
venv_path=""
dataset_path=""

usage() {
    cat <<EOF
Usage: $0 <engine> [--venv-path PATH] [--port PORT] [--model MODEL_ID] [--dataset-path PATH]

Positional arguments:
  engine         Target engine (vllm | sglang) [required]
Options:
  --venv-path    Path to a virtual environment to activate (optional)
  --port         Server port (default: vllm=$DEFAULT_PORT_VLLM, sglang=$DEFAULT_PORT_SGL)
  --model        Model identifier (default: $DEFAULT_MODEL)
  --dataset-path Path to local ShareGPT JSON file (optional, auto-downloads if omitted)
  -h, --help     Show this help and exit

Example:
  $0 vllm --venv-path ../../engine_integration/vllm-pip-venv --model meta-llama/Llama-3.2-1B
  $0 vllm --model meta-llama/Llama-3.2-1B --dataset-path /root/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
EOF
}

# Parse long options via getopt
TEMP=$(getopt \
    --options h \
    --longoptions port:,model:,venv-path:,dataset-path:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then exit 1; fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --port) port="$2"; shift 2;;
        --model) model="$2"; shift 2;;
        --venv-path) venv_path="$2"; shift 2;;
        --dataset-path) dataset_path="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        --) shift; break;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Positional engine arg
if [[ $# -lt 1 ]]; then echo "Error: engine positional argument required" >&2; usage; exit 1; fi
engine="$1"; shift

# Validate engine
if [[ "$engine" != "vllm" && "$engine" != "sglang" ]]; then
    echo "Error: engine must be 'vllm' or 'sglang'" >&2; usage; exit 1
fi

# Validate venv_path if supplied
if [[ -n "$venv_path" ]]; then
    if [[ ! -f "$venv_path/bin/activate" ]]; then
        echo "Error: --venv-path '$venv_path' is invalid (activate script not found)" >&2; exit 1
    fi
fi

# Apply defaults
MODEL=${model:-$DEFAULT_MODEL}
if [[ -n "$port" ]]; then
    ENGINE_PORT=$port
else
    if [[ "$engine" == "vllm" ]]; then
        ENGINE_PORT=$DEFAULT_PORT_VLLM;
    else
        ENGINE_PORT=$DEFAULT_PORT_SGL;
    fi
fi
if [[ "$engine" == "vllm" ]]; then
    VLLM_PORT=$ENGINE_PORT;
else
    SGL_PORT=$ENGINE_PORT;
fi

PYTHON=${PYTHON:-python3}

source "$SCRIPT_DIR/env_detect.sh"

run_auto_plot_if_enabled() {
    local plot_flag="${KVCACHED_AUTO_PLOT:-0}"
    if [[ "$plot_flag" != "1" && "${plot_flag,,}" != "true" ]]; then
        return 0
    fi

    local wait_s="${KVCACHED_PLOT_WAIT_SECONDS:-6}"
    local report_json="${KVCACHED_FREE_REPORT_JSON:-$KVCACHED_DIR/kvcached_free_debug_report.json}"
    local out_dir="${KVCACHED_PLOT_OUT_DIR:-$KVCACHED_DIR/kvcached_free_plots}"
    local prefix="${KVCACHED_PLOT_PREFIX:-kvcached_free}"
    local plot_script="$KVCACHED_DIR/tools/kvcached_free_plot.py"

    echo "Waiting ${wait_s}s for idle-report JSON flush..."
    sleep "$wait_s"

    if [[ ! -f "$report_json" ]]; then
        echo "Skip plotting: report JSON not found at $report_json"
        return 0
    fi
    if [[ ! -f "$plot_script" ]]; then
        echo "Skip plotting: plot script not found at $plot_script"
        return 0
    fi

    echo "Generating kvcached plots from $report_json ..."
    $PYTHON "$plot_script" \
        --report-json "$report_json" \
        --out-dir "$out_dir" \
        --prefix "$prefix"
}

# Resolve dataset path: use --dataset-path if provided, otherwise auto-download
if [[ -n "$dataset_path" ]]; then
    if [[ ! -f "$dataset_path" ]]; then
        echo "Error: --dataset-path '$dataset_path' does not exist" >&2; exit 1
    fi
    DATASET_FILE="$dataset_path"
else
    # Auto-download to SCRIPT_DIR if not present
    DATASET_FILE="$SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
    if [ ! -f "$DATASET_FILE" ]; then
        pushd "$SCRIPT_DIR"
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
        popd
    fi
fi

if [ "$engine" == "vllm" ]; then
    if [[ -n "$venv_path" ]]; then source "$venv_path/bin/activate"; fi
    vllm bench serve \
      --model $MODEL \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_FILE" \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --port $VLLM_PORT
    run_auto_plot_if_enabled
    if [[ -n "$venv_path" ]]; then deactivate; fi
elif [ "$engine" == "sgl" -o "$engine" == "sglang" ]; then
    if [[ -n "$venv_path" ]]; then source "$venv_path/bin/activate"; fi

    $PYTHON -m sglang.bench_serving --backend sglang-oai \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_FILE" \
        --request-rate $REQUEST_RATE \
        --num-prompts $NUM_PROMPTS \
        --port $SGL_PORT
    run_auto_plot_if_enabled
    if [[ -n "$venv_path" ]]; then deactivate; fi
else
    echo "Invalid engine: $engine"
    exit 1
fi
