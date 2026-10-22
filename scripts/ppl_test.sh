#!/bin/bash
# PPL Test Suite for OSCAR2 / KV-cache comparison
# Constraint: oscar2 can only mix with f16/oscar2 or oscar2/oscar2 (not q8_0)
# Compare against: f16/f16 (baseline) and q8_0/turbo4 (comparison)
#
# Protocol (from oscar2-llama-debug skill):
#   -c <ctx> --chunks -1 --no-warmup -ctk <K> -ctv <V> -fa on -ngl 0 -t 32 -f <dataset>
# Using -c 512 --chunks 1 for speed (full ctx=4096 chunks=-1 takes 5+ min per run)

set -euo pipefail

ROOT="${LLAMA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CLI="${PPL_CLI:-${ROOT}/build/bin/llama-perplexity}"
LIB="${PPL_LIB:-${ROOT}/build/bin}"
export LD_LIBRARY_PATH="${LIB}:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="${PPL_DATASET:?set PPL_DATASET to a plain-text perplexity dataset (for example wiki.test.raw)}"
OUTDIR="${PPL_OUTDIR:-${ROOT}/ppl_results}"
mkdir -p "$OUTDIR"

CTX="${PPL_CTX:-512}"
CHUNKS="${PPL_CHUNKS:-1}"
THREADS="${PPL_THREADS:-$(nproc)}"

TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="${OUTDIR}/ppl_summary_${TS}.tsv"
LOGDIR="${OUTDIR}/logs_${TS}"
mkdir -p "$LOGDIR"

echo -e "TEST_ID\tModel\tConfig\tCTK\tCTV\tPPL\tLogTps\tEvalTps\tCacheInfo\tExitCode" > "$SUMMARY"

run_ppl() {
    local test_id="$1"
    local model_name="$2"
    local model_path="$3"
    local ctk="$4"
    local ctv="$5"
    shift 5

    local logfile="${LOGDIR}/test_${test_id}.log"
    echo ""
    echo "======================================================================"
    echo "  TEST ${test_id}: ${model_name} (KV K=${ctk}, V=${ctv})"
    echo "  Log: ${logfile}"
    echo "======================================================================"

    # Split KEY=value args (environment overrides) from regular CLI args.
    local env_vars=()
    local extra_args=()
    for arg in "$@"; do
        if [[ "$arg" == *=* ]]; then
            env_vars+=("$arg")
        else
            extra_args+=("$arg")
        fi
    done

    # Run with environment overrides and any remaining CLI args.
    env -i PATH="/usr/bin:/bin:/usr/local/bin" \
        LD_LIBRARY_PATH="${LIB}:${LD_LIBRARY_PATH:-}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        OMP_NUM_THREADS="$THREADS" \
        ${env_vars[@]+"${env_vars[@]}"} \
        "$CLI" \
        -m "$model_path" \
        -c "$CTX" \
        --chunks "$CHUNKS" \
        --no-warmup \
        -ctk "$ctk" \
        -ctv "$ctv" \
        -fa on \
        -ngl 0 \
        -t "$THREADS" \
        -f "$DATASET" \
        ${extra_args[@]+"${extra_args[@]}"} \
        2>&1 | tee "$logfile"

    local rc=${PIPESTATUS[0]}

    if [ $rc -ne 0 ]; then
        echo "  EXIT CODE: $rc"
        echo -e "${test_id}\t${model_name}\t${ctk}/${ctv}\t${ctk}\t${ctv}\tFAILED(exit:${rc})\t\t\t\t${rc}" >> "$SUMMARY"
        return
    fi

    # Parse PPL from log
    local ppl=$(grep -i "Final estimate" "$logfile" | grep -oP 'PPL = \K[0-9.]+' || echo "N/A")
    if [ "$ppl" = "N/A" ]; then
        ppl=$(grep -i "perplexity" "$logfile" | tail -1 | grep -oP 'PPL = \K[0-9.]+' || echo "N/A")
    fi

    local log_tps=$(grep -oP 'Prompt:\s*\K[0-9.]+' "$logfile" | head -1 || echo "N/A")
    local eval_tps=$(grep -oP 'Generation:\s*\K[0-9.]+' "$logfile" | tail -1 || echo "N/A")

    # Also try llama_perf_context_eval format
    if [ "$eval_tps" = "N/A" ]; then
        eval_tps=$(grep "llama_perf_context_eval" "$logfile" | tail -1 | grep -oP 't/s.*\K[0-9.]+' || echo "N/A")
    fi

    # Extract cache info
    local cache_info=$(grep -i "KV cache.*dtype" "$logfile" | tail -1 | grep -oP 'K=\S+ V=\S+' || echo "")

    echo "  Result: PPL=${ppl}, Cache=${cache_info}"
    echo -e "${test_id}\t${model_name}\t${ctk}/${ctv}\t${ctk}\t${ctv}\t${ppl}\t${log_tps}\t${eval_tps}\t${cache_info}\t${rc}" >> "$SUMMARY"
}

# ============================================================
# Model definitions
# ============================================================

MODELS_DIR="${PPL_MODELS_DIR:?set PPL_MODELS_DIR to the directory containing the OSCAR rotation GGUF models}"
KW_DIR="${PPL_KW_DIR:-${MODELS_DIR}}"

GEMMA12B="${PPL_GEMMA12B:-${MODELS_DIR}/gemma-4-12b-it-UD-Q8_K_XL-rot-kv.gguf}"
GEMMA26B="${PPL_GEMMA26B:-${MODELS_DIR}/gemma-4-26B-A4B-it-UD-Q5_K_S-rot-kv.gguf}"
QWEN27B="${PPL_QWEN27B:-${MODELS_DIR}/Qwen3.6-27B-UD-Q5_K_XL-rot-kv.gguf}"
KWAIPLOT="${PPL_KWAIPLOT:-${KW_DIR}/Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_S.gguf}"

echo "================================================================="
echo "  PPL Test Suite Started: $(date)"
echo "  Dataset: ${DATASET}"
echo "  Context: ${CTX}, Chunks: ${CHUNKS}, Threads: ${THREADS}"
echo "  Build: ${ROOT}"
echo "================================================================="

# ============================================================
# Test matrix:
#   f16/f16 (baseline)       — all 4 models
#   oscar2/oscar2            — 3 rot-kv models (Qwen27B, Gemma12B, Gemma26B)
#   oscar2/f16               — 3 rot-kv models (allowed mix)
#   f16/oscar2               — 3 rot-kv models (allowed mix)
#   q8_0/q8_0                — all 4 models
#   q8_0/turbo4              — all 4 models
#   f16/oscar2 on Kwaipilot  — expected to fail (no rot-kv)
# ============================================================

echo "=== Phase 1: f16/f16 baselines ==="

run_ppl "g12_f16_f16" "gemma4-12b" "$GEMMA12B" "f16" "f16"
run_ppl "g26_f16_f16" "gemma4-26b-a4b" "$GEMMA26B" "f16" "f16"
run_ppl "q27_f16_f16" "qwen3.6-27b" "$QWEN27B" "f16" "f16"
run_ppl "kw_f16_f16" "kwaipilot-35b" "$KWAIPLOT" "f16" "f16"

echo ""
echo "=== Phase 2: oscar2/oscar2 on rot-kv models ==="

run_ppl "g12_oscar2_oscar2" "gemma4-12b" "$GEMMA12B" "oscar2" "oscar2"
run_ppl "g26_oscar2_oscar2" "gemma4-26b-a4b" "$GEMMA26B" "oscar2" "oscar2"
run_ppl "q27_oscar2_oscar2" "qwen3.6-27b" "$QWEN27B" "oscar2" "oscar2"

echo ""
echo "=== Phase 3: oscar2/f16 mix on rot-kv models ==="

run_ppl "g12_oscar2_f16" "gemma4-12b" "$GEMMA12B" "oscar2" "f16"
run_ppl "g26_oscar2_f16" "gemma4-26b-a4b" "$GEMMA26B" "oscar2" "f16"
run_ppl "q27_oscar2_f16" "qwen3.6-27b" "$QWEN27B" "oscar2" "f16"

echo ""
echo "=== Phase 4: f16/oscar2 mix on rot-kv models ==="

run_ppl "g12_f16_oscar2" "gemma4-12b" "$GEMMA12B" "f16" "oscar2"
run_ppl "g26_f16_oscar2" "gemma4-26b-a4b" "$GEMMA26B" "f16" "oscar2"
run_ppl "q27_f16_oscar2" "qwen3.6-27b" "$QWEN27B" "f16" "oscar2"

echo ""
echo "=== Phase 5: q8_0/q8_0 on all models ==="

run_ppl "g12_q8_0_q8_0" "gemma4-12b" "$GEMMA12B" "q8_0" "q8_0"
run_ppl "g26_q8_0_q8_0" "gemma4-26b-a4b" "$GEMMA26B" "q8_0" "q8_0"
run_ppl "q27_q8_0_q8_0" "qwen3.6-27b" "$QWEN27B" "q8_0" "q8_0"
run_ppl "kw_q8_0_q8_0" "kwaipilot-35b" "$KWAIPLOT" "q8_0" "q8_0"

echo ""
echo "=== Phase 6: q8_0/turbo4 on all models ==="

run_ppl "g12_q8_0_turbo4" "gemma4-12b" "$GEMMA12B" "q8_0" "turbo4"
run_ppl "g26_q8_0_turbo4" "gemma4-26b-a4b" "$GEMMA26B" "q8_0" "turbo4"
run_ppl "q27_q8_0_turbo4" "qwen3.6-27b" "$QWEN27B" "q8_0" "turbo4"
run_ppl "kw_q8_0_turbo4" "kwaipilot-35b" "$KWAIPLOT" "q8_0" "turbo4"

echo ""
echo "=== Phase 7: oscar2/oscar2 on Kwaipilot (expected to fail - no rot-kv) ==="

run_ppl "kw_oscar2_oscar2" "kwaipilot-35b" "$KWAIPLOT" "oscar2" "oscar2"

echo ""
echo "======================================================================"
echo "  All PPL tests complete! $(date)"
echo "  Summary: ${SUMMARY}"
echo "======================================================================"
echo ""
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"