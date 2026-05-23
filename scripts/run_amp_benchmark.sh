#!/bin/bash
# scripts/b200/run_amp_benchmark.sh
# AMP ON/OFF A/B + 3-epoch convergence benchmark, designed for B200 x2 SECURE Pod.
#
# Designed to run INSIDE the v5 image container (Pod). Host (u58) collects results via scp.
#
# Runs:
#   A: bevformer_tiny_amp_fp32.py, 1 epoch, AMP OFF, optional Nsight Compute (ncu)
#   B: bevformer_tiny_amp_fp16.py, 1 epoch, AMP ON,  optional Nsight Compute (ncu)
#   C: bevformer_tiny_amp_fp16.py, 3 epochs, AMP ON, no ncu (loss curve only)
#
# Outputs (per run): /workspace/work_dirs/amp_bench/{a_fp32,b_fp16,c_fp16_3ep}/
#   - $RUN_DIR/train.log               (mmdet/mmcv train.py stdout)
#   - $RUN_DIR/iter_times.txt          (parsed per-iter timings)
#   - $RUN_DIR/ncu_report.ncu-rep      (if ncu enabled and target is A or B)
#   - $RUN_DIR/loss_curve.tsv          (parsed loss progression for C)
#
# Usage:
#   bash run_amp_benchmark.sh [--runs A,B,C] [--skip-ncu] [--config-root /workspace/bevformer/projects/configs/bevformer]
#
# Pre-conditions (Pod 側で別途確認):
#   - nuScenes-mini が /workspace/data/nuScenes-mini/ にある (or 別 path で --data-root override)
#   - mini 用 pkl (nuscenes_infos_temporal_{train,val}_mini.pkl) が同 dir にある
#   - ncu が PATH にある (nvidia-cuda-toolkit / nvidia-nsight-compute install)
#   - GPU: 2 x B200 visible (nvidia-smi で確認)
#
set -uo pipefail

# ============================================================
# Defaults
# ============================================================
RUNS_DEFAULT="A,B,C"
SKIP_NCU=0
# Paths point at baked image locations (release image v6 / blackwell-pt2.8).
# Override at invocation time via --config-root / --data-root if Pod layout differs.
CONFIG_ROOT="/opt/bevformer/projects/configs/bevformer"
DATA_ROOT="/workspace/data/nuscenes"
WORKDIR_BASE="/workspace/work_dirs/amp_bench"
TRAIN_PY="/opt/bevformer/tools/train.py"
DIST_TRAIN_SH="/opt/bevformer/tools/dist_train.sh"
GPUS=2

# PYTHONPATH must include /opt/bevformer so the plugin is importable from
# anywhere — Cmd 1 verified pattern.
export PYTHONPATH="/opt/bevformer:${PYTHONPATH:-}"

# Ncu metrics for Tensor Core utilization (主目的)
NCU_METRICS=(
  "sm__inst_executed_pipe_tensor.sum"           # Tensor Core instr 総数
  "sm__cycles_elapsed.sum"                       # 全 SM cycle
  "sm__cycles_active.sum"                        # 動作 SM cycle
  "sm__inst_executed.sum"                        # 全 inst
  "dram__bytes.sum"                              # HBM 帯域
  "gpu__time_duration.sum"                       # GPU active time
)
NCU_KERNEL_FILTER=""  # 例: --kernel-name "ms_deform_attn|gemm" で絞る、空ならフル

# ============================================================
# Args parsing
# ============================================================
RUNS_SEL="$RUNS_DEFAULT"
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS_SEL="$2"; shift 2;;
    --skip-ncu) SKIP_NCU=1; shift;;
    --config-root) CONFIG_ROOT="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    -h|--help) sed -n '1,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

# ============================================================
# Pre-flight: env + data verification
# ============================================================
echo "=== Pre-flight checks ==="
echo "GPUs visible:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 | head -5
echo
echo "ncu (Nsight Compute) availability:"
if command -v ncu >/dev/null 2>&1; then
  ncu --version 2>&1 | head -2
  NCU_AVAILABLE=1
else
  echo "  ncu NOT installed → ncu wrapping disabled"
  NCU_AVAILABLE=0
fi
[ "$SKIP_NCU" -eq 1 ] && NCU_AVAILABLE=0

echo
echo "Dataset path: $DATA_ROOT"
if [ -d "$DATA_ROOT" ]; then
  echo "  dir present"
  for required in "v1.0-trainval" "samples" "nuscenes_infos_temporal_train.pkl" "nuscenes_infos_temporal_val.pkl"; do
    if [ -e "$DATA_ROOT/$required" ]; then
      echo "  [OK] $required"
    else
      echo "  [MISSING] $required"
    fi
  done
else
  echo "  [FAIL] $DATA_ROOT not found — generate info pkls first:"
  echo "    cd /workspace && python /opt/bevformer/tools/create_data.py nuscenes \\"
  echo "      --root-path $DATA_ROOT --out-dir $DATA_ROOT \\"
  echo "      --extra-tag nuscenes --version v1.0 --canbus /workspace/data/"
  exit 3
fi
echo
echo "Configs:"
for cfg in bevformer_tiny_amp_fp32.py bevformer_tiny_amp_fp16.py; do
  if [ -f "$CONFIG_ROOT/$cfg" ]; then
    echo "  [OK] $CONFIG_ROOT/$cfg"
  else
    echo "  [FAIL] $CONFIG_ROOT/$cfg not found"
    exit 4
  fi
done
echo

# ============================================================
# Helpers
# ============================================================
parse_iter_times() {
  local log="$1"
  # mmdet log format: "Epoch [1][10/N] ... time: 0.523 ..."
  grep -oE "time: [0-9]+\.[0-9]+" "$log" 2>/dev/null | awk '{print $2}' > "${log%.log}_iter_times.txt"
  local n=$(wc -l < "${log%.log}_iter_times.txt")
  if [ "$n" -gt 0 ]; then
    local mean=$(awk 'BEGIN{s=0; n=0} {s+=$1; n++} END{if(n>0) printf "%.4f", s/n}' "${log%.log}_iter_times.txt")
    local p50=$(sort -n "${log%.log}_iter_times.txt" | awk -v n="$n" 'NR==int(n/2+1){print}')
    echo "  iter time: n=$n mean=${mean}s p50=${p50}s"
  fi
}

parse_loss_curve() {
  local log="$1"
  # mmdet log: "Epoch [1][10/N] ... loss: 9.876 ..."
  grep -oE "Epoch \[[0-9]+\]\[[0-9]+/[0-9]+\].*loss: [0-9]+\.[0-9]+" "$log" 2>/dev/null \
    | awk '{
        for (i=1; i<=NF; i++) {
          if ($i == "loss:") { loss=$(i+1); }
        }
        # extract epoch+iter
        match($0, /\[[0-9]+\]\[[0-9]+\/[0-9]+\]/); s=substr($0, RSTART, RLENGTH);
        print s "\t" loss;
      }' > "${log%.log}_loss_curve.tsv"
  local n=$(wc -l < "${log%.log}_loss_curve.tsv")
  echo "  loss curve: $n data points"
  if [ "$n" -gt 2 ]; then
    echo "  first 3: $(head -3 ${log%.log}_loss_curve.tsv | tr '\n' ' | ')"
    echo "  last 3:  $(tail -3 ${log%.log}_loss_curve.tsv | tr '\n' ' | ')"
  fi
}

# ============================================================
# Run executor
# ============================================================
run_single() {
  local LABEL="$1"      # a_fp32 / b_fp16 / c_fp16_3ep
  local CONFIG="$2"     # full path to .py
  local EPOCHS="$3"     # 1 or 3
  local NCU_ENABLED="$4"  # 0/1

  local RUN_DIR="$WORKDIR_BASE/$LABEL"
  mkdir -p "$RUN_DIR"
  local LOGFILE="$RUN_DIR/train.log"

  echo
  echo "=== Run $LABEL — config=$(basename $CONFIG) epochs=$EPOCHS ncu=$NCU_ENABLED ==="
  echo "  output: $RUN_DIR"
  local START=$(date +%s)

  # Build base train command. Note: dist_train.sh / config / work-dir are all
  # absolute paths; --no-validate skips eval (Phase 4 で確定の DataContainer skip
  # 経路 — full data + AMP benchmark には eval 不要)。
  local TRAIN_CMD=(
    "$DIST_TRAIN_SH" "$CONFIG" "$GPUS"
    --no-validate
    --work-dir "$RUN_DIR"
    --cfg-options "total_epochs=$EPOCHS" "runner.max_epochs=$EPOCHS"
  )

  # cwd MUST be /workspace because bevformer_tiny.py uses `data_root='data/nuscenes/'`
  # (relative). Cmd 1 (training) was verified with this pattern.
  pushd /workspace > /dev/null

  if [ "$NCU_ENABLED" -eq 1 ] && [ "$NCU_AVAILABLE" -eq 1 ]; then
    # ncu wrapping: profile launches but limited to first few kernels to avoid huge slowdown
    local METRICS_CSV=$(IFS=,; echo "${NCU_METRICS[*]}")
    local NCU_REP="$RUN_DIR/ncu_report.ncu-rep"
    echo "  ncu enabled, report → $NCU_REP"
    ncu --target-processes all \
        --replay-mode application \
        --metrics "$METRICS_CSV" \
        --launch-skip 50 \
        --launch-count 200 \
        --export "$NCU_REP" \
        --force-overwrite \
        bash "${TRAIN_CMD[@]}" 2>&1 | tee "$LOGFILE"
  else
    bash "${TRAIN_CMD[@]}" 2>&1 | tee "$LOGFILE"
  fi

  popd > /dev/null

  local ELAPSED=$(( $(date +%s) - START ))
  echo "  elapsed: ${ELAPSED}s"
  parse_iter_times "$LOGFILE"
  parse_loss_curve "$LOGFILE"
}

# ============================================================
# Main: run selected runs
# ============================================================
mkdir -p "$WORKDIR_BASE"

if echo ",$RUNS_SEL," | grep -q ",A,"; then
  run_single "a_fp32"     "$CONFIG_ROOT/bevformer_tiny_amp_fp32.py" 1 1
fi
if echo ",$RUNS_SEL," | grep -q ",B,"; then
  run_single "b_fp16"     "$CONFIG_ROOT/bevformer_tiny_amp_fp16.py" 1 1
fi
if echo ",$RUNS_SEL," | grep -q ",C,"; then
  run_single "c_fp16_3ep" "$CONFIG_ROOT/bevformer_tiny_amp_fp16.py" 3 0
fi

echo
echo "=== All runs done ==="
echo "Results under: $WORKDIR_BASE"
ls -la "$WORKDIR_BASE/" 2>/dev/null
echo
echo "To collect on host:"
echo "  scp -r -i ~/.runpod/ssh/RunPod-Key-Go root@<pod-host>:$WORKDIR_BASE /tmp/amp_bench_results"
