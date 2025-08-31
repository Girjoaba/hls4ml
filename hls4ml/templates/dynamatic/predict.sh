#!/bin/bash
set -e

DYNAMATIC_PATH=$1
F_SRC=$2
FUNC_NAME=$3
OUT="./out-$FUNC_NAME"

bash "../csim.sh" \
  "$DYNAMATIC_PATH" \
  "$F_SRC" \
  "$FUNC_NAME" \
  "$OUT" \
  "$OUT/sim"