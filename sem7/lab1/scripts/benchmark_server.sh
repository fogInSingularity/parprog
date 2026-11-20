#!/usr/bin/env bash
# Run manually on the server; this script is never part of CTest.
set -euo pipefail

matrix_binary=${1:-./build/matrix_lab}
matrix_results=${2:-./results/server}
mkdir -p "$matrix_results"

{
    date -u
    uname -a
    if command -v lscpu >/dev/null 2>&1; then
        lscpu
    fi
    "$matrix_binary" info
} > "$matrix_results/environment.txt"

matrix_build_dir=$(dirname "$matrix_binary")
if [[ -f "$matrix_build_dir/compile_commands.json" ]]; then
    cp "$matrix_build_dir/compile_commands.json" "$matrix_results/compile_commands.json"
fi

"$matrix_binary" admission --size 512 > "$matrix_results/admission.txt"

matrix_csv=$(mktemp "$matrix_results/.raw.XXXXXX")
trap 'rm -f -- "$matrix_csv" "$matrix_csv.meta.txt"' EXIT

matrix_benchmark_status=0
"$matrix_binary" bench \
    --sizes "${MATRIX_SIZES:-32,64,128,256,512,1024,2048}" \
    --threads "${MATRIX_THREADS:-1,2,8,32}" \
    --values "${MATRIX_VALUES:-positive}" \
    --block "${MATRIX_BLOCK:-32}" --cutoff "${MATRIX_CUTOFF:-64}" \
    --repeats "${MATRIX_REPEATS:-5}" --warmups "${MATRIX_WARMUPS:-1}" \
    --output "$matrix_csv" | tee "$matrix_results/run.txt" || matrix_benchmark_status=$?

if [[ ! -f "$matrix_csv.meta.txt" ]] || (( $(wc -l < "$matrix_csv") < 2 )); then
    printf 'The benchmark produced no validated measurements.\n' >&2
    exit 1
fi

mv -- "$matrix_csv" "$matrix_results/raw.csv"
mv -- "$matrix_csv.meta.txt" "$matrix_results/raw.csv.meta.txt"
printf 'Measurements saved to %s\n' "$matrix_results/raw.csv"

python3 "$(dirname "$0")/plot.py" "$matrix_results/raw.csv" \
    --output "$matrix_results/plots" --title "Server: matrix multiplication, float32"

exit "$matrix_benchmark_status"
