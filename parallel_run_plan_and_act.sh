#!/bin/bash

################################################################################
# USAGE:
#   ./parallel_run_plan_and_act.sh <TOTAL_PANES>
#
# Dynamically creates the specified number of panes and distributes the workload.
################################################################################

# -------------------------- MODIFY THESE IF NEEDED ----------------------------
test_config_base_dir='config_files/wa/test_webarena'
precomputed_cot_plans_path="/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot_plans/webarena_cot_plans_after_10000.json"
actor_ip="http://localhost:8000/v1"
cot_actor_model="/home/lerdogan/torchtune/output/deepseek-r1-70B-full-executor/full/epoch_0"
max_tokens=4096
result_dir="/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/eval_results/cot_plans_10000_full_webarena"
max_steps=30

SERVER='35.188.138.44'
MAP_SERVER='35.188.138.44'
OPENAI_API_KEY='webarena'
OPENAI_ORGANIZATION=''
CONDA_ENV_NAME='visualwebarena'

ENV_VARIABLES="export TIKTOKEN_CACHE_DIR=""; export DATASET=${DATASET}; export SHOPPING='http://${SERVER}:8082'; export SHOPPING_ADMIN='http://${SERVER}:8083/admin'; export REDDIT='http://${SERVER}:8080'; export GITLAB='http://${SERVER}:9001'; export MAP='http://${MAP_SERVER}:443'; export WIKIPEDIA='http://${SERVER}:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'; export HOMEPAGE='http://${SERVER}:4399'; export OPENAI_API_KEY=${OPENAI_API_KEY}; export OPENAI_ORGANIZATION=${OPENAI_ORGANIZATION}"

TOLERANCE=2
TOTAL_EXAMPLES=812

################################################################################
# 1) Parse command-line argument for desired number of total panes
################################################################################
if [ $# -lt 1 ]; then
  echo "Usage: $0 <TOTAL_PANES>"
  echo "Example: $0 8 -> 1 monitor + 7 job panes."
  exit 1
fi

desired_panes=$1

# Count existing panes
num_panes=$(tmux list-panes | wc -l)

# Calculate how many new panes need to be created
panes_to_create=$((desired_panes - num_panes))

if [ $panes_to_create -gt 0 ]; then
    for ((i=0; i<$panes_to_create; i++)); do
        tmux split-window -v
        tmux select-layout tiled
    done
elif [ $panes_to_create -lt 0 ]; then
    echo "Already have $num_panes panes, which is more than desired. No new panes created."
else
    echo "Exactly $desired_panes panes. Nothing to do."
fi

################################################################################
# 2) Helper Functions
################################################################################
generate_intervals() {
    local total_examples=$1
    local num_jobs=$2

    local intervals=()
    local quotient=$((total_examples / num_jobs))
    local remainder=$((total_examples % num_jobs))

    local start=0
    local end=0

    for (( i=0; i<${num_jobs}; i++ )); do
        chunk_size=${quotient}
        if [ $i -lt ${remainder} ]; then
            chunk_size=$((quotient + 1))
        fi

        end=$((start + chunk_size))
        if [ $end -gt $total_examples ]; then
            end=$total_examples
        fi

        intervals+=( "$start" )
        intervals+=( "$end" )

        start=$end
    done

    echo "${intervals[@]}"
}

run_job() {
    local pane_index=$1
    local start_idx=$2
    local end_idx=$3

    tmux select-pane -t "${pane_index}"

    COMMAND="python run_plan_and_act.py \
        --test_start_idx ${start_idx} \
        --test_end_idx ${end_idx} \
        --test_config_base_dir ${test_config_base_dir} \
        --precomputed_cot_plans_path ${precomputed_cot_plans_path} \
        --viewport_width 1280 \
        --viewport_height 720 \
        --actor_ip ${actor_ip} \
        --cot_actor_model ${cot_actor_model} \
        --max_tokens ${max_tokens} \
        --max_steps ${max_steps} \
        --result_dir ${result_dir} \
        --action_set_tag webrl_id \
        --observation_type webrl \
        --current_viewport_only"

    tmux send-keys "tmux set mouse on; source ~/.bashrc; conda activate ${CONDA_ENV_NAME}; ${ENV_VARIABLES}; until ${COMMAND}; do echo 'crashed' >&2; sleep 1; done" C-m
    sleep 3
}

run_batch() {
    local args=("$@")
    local num_args=${#args[@]}

    if (( num_args % 2 != 0 )); then
        echo "ERROR: run_batch requires an even number of arguments (start/end pairs)."
        return 1
    fi

    local job_index=1  # Skip pane 0 (monitor)

    for ((i=0; i<$num_args; i+=2)); do
        local start_idx=${args[i]}
        local end_idx=${args[i+1]}

        run_job $job_index $start_idx $end_idx

        ((job_index++))
    done

    echo "Waiting for all python processes to finish..."
    while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
        sleep 10
    done

    # Check for errors and rerun if needed
    if [ -f "scripts/check_error_runs.py" ]; then
        while ! python scripts/check_error_runs.py ${result_dir} --delete_errors --tolerance ${TOLERANCE}; do
            echo "Check failed, re-running failed jobs..."

            job_index=1
            for ((i=0; i<$num_args; i+=2)); do
                local start_idx=${args[i]}
                local end_idx=${args[i+1]}
                run_job $job_index $start_idx $end_idx
                ((job_index++))
            done

            while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
                sleep 10
            done
        done
    else
        echo "Warning: scripts/check_error_runs.py not found. Skipping error checking."
    fi

    echo "All jobs completed successfully!"
}

################################################################################
# 3) Main Script Execution
################################################################################

# Calculate number of job panes (total panes minus monitor pane)
let num_jobs=desired_panes-1
if [ $num_jobs -le 0 ]; then
    echo "ERROR: You must specify at least 2 total panes (1 monitor + 1 job)."
    exit 1
fi

# Generate intervals
intervals_array=( $(generate_intervals $TOTAL_EXAMPLES $num_jobs) )
echo "Intervals for $TOTAL_EXAMPLES examples in $num_jobs job panes:"
echo "${intervals_array[@]}"

# Create the main result directory
mkdir -p "${result_dir}"
mkdir -p "${result_dir}/actions"
mkdir -p "${result_dir}/traces"
mkdir -p "${result_dir}/screehshots"

# Run batch
run_batch "${intervals_array[@]}"
