#!/usr/bin/env bash
set -euo pipefail
IFS=, read -r -a cpus <<< "$PINNED_CPUS"
rank=${OMPI_COMM_WORLD_LOCAL_RANK:?}
cpu=${cpus[$rank]:?}
taskset -pc "$cpu" "$$" >/dev/null
actual=$(awk '/^Cpus_allowed_list:/ { print $2 }' "/proc/$$/status")
[[ "$actual" == "$cpu" ]]
printf 'MPI affinity: rank=%s cpu=%s\n' "$rank" "$actual"
exec "$@"
