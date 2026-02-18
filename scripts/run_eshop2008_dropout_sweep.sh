#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="eshop2008"
RHO_VALUES=("0.15" "0.20" "0.25" "0.30")
SUMMARY_CSV="results/eshop2008_dropout_sweep_summary.csv"

mkdir -p results
echo "architecture,item_dropout,best_valid_recall20,best_valid_mrr20,test_recall20,test_mrr20,log_file" > "$SUMMARY_CSV"

run_one() {
  local arch_name="$1"
  local run_dir="$2"
  local model="$3"
  local log_dir_rel="$4"
  local rho="$5"

  local py_bin cmd log_dir latest_log dest_log lines best_line test_line
  local best_recall best_mrr test_recall test_mrr

  if [[ "$run_dir" == "." ]]; then
    py_bin=".venv/bin/python"
    cmd="$py_bin main.py --model $model --dataset $DATASET --item-dropout $rho"
  else
    py_bin="../.venv/bin/python"
    cmd="$py_bin main.py --model $model --dataset $DATASET --item-dropout $rho"
  fi

  echo "[$(date '+%F %T')] START arch=$arch_name rho=$rho"
  (cd "$run_dir" && eval "$cmd")

  log_dir="$ROOT_DIR/$log_dir_rel"
  latest_log="$(ls -1t "$log_dir"/*.log | head -n 1)"
  dest_log="results/run_${arch_name}_${DATASET}_dropout_${rho}.log"
  cp -f "$latest_log" "$dest_log"

  lines="$(rg -n "best valid|test result" "$dest_log" | tail -n 2)"
  best_line="$(echo "$lines" | sed -n '1p')"
  test_line="$(echo "$lines" | sed -n '2p')"

  best_recall="$(echo "$best_line" | sed -E "s/.*\\('recall@20', ([0-9.]+)\\).*\\('mrr@20', ([0-9.]+)\\).*/\\1/")"
  best_mrr="$(echo "$best_line" | sed -E "s/.*\\('recall@20', ([0-9.]+)\\).*\\('mrr@20', ([0-9.]+)\\).*/\\2/")"
  test_recall="$(echo "$test_line" | sed -E "s/.*\\('recall@20', ([0-9.]+)\\).*\\('mrr@20', ([0-9.]+)\\).*/\\1/")"
  test_mrr="$(echo "$test_line" | sed -E "s/.*\\('recall@20', ([0-9.]+)\\).*\\('mrr@20', ([0-9.]+)\\).*/\\2/")"

  echo "$arch_name,$rho,$best_recall,$best_mrr,$test_recall,$test_mrr,$dest_log" >> "$SUMMARY_CSV"
  echo "[$(date '+%F %T')] DONE  arch=$arch_name rho=$rho test_recall=$test_recall test_mrr=$test_mrr"
}

for rho in "${RHO_VALUES[@]}"; do
  run_one "core_trm" "." "trm" "log/COREtrm" "$rho"
  run_one "dual_attention" "Dual-Attention" "trm_da" "Dual-Attention/log/COREtrmDualAttention" "$rho"
  run_one "hard_negatives" "Hard Negatives" "trm" "Hard Negatives/log/COREtrm" "$rho"
  run_one "dual_attention_hard_negatives" "Dual-Attention-Hard-Negatives" "trm_da" "Dual-Attention-Hard-Negatives/log/COREtrmDualAttention" "$rho"
done

echo
echo "Sweep complete. Summary written to: $SUMMARY_CSV"
