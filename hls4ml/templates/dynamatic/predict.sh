#!/bin/bash
set -e

FUNC_NAME=$1
OUT="./out-$FUNC_NAME"

"$OUT/sim/output_gen"

# bash "../csim.sh" \
#   "$DYNAMATIC_PATH" \
#   "$F_SRC" \
#   "$FUNC_NAME" \
#   "$OUT" \
#   "$OUT/sim"