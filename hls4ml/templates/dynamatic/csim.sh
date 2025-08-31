#!/bin/bash

DYNAMATIC=$1
SOURCE="$2"
KERNEL_NAME="$3"
OUTPUT_DIR=$(realpath "$4")
SIM_DIR=$(realpath "$5")

# rm -rf "$SIM_DIR/"

# mkdir -p "$SIM_DIR/"{C_SRC,C_OUT,INPUT_VECTORS}

# mkdir -p "$SIM_DIR/C_SRC/dynamatic/"

# cp "$DYNAMATIC/include/dynamatic/Integration.h" \
#   "$SIM_DIR/C_SRC/dynamatic/"

# cp "$SOURCE" "$SIM_DIR/C_SRC"


# # Compile the source with verification flags
# "$DYNAMATIC/polygeist/llvm-project/build/bin/clang++" \
#   "$SOURCE" \
#   -D HLS_VERIFICATION \
#   -DHLS_VERIFICATION_PATH="$SIM_DIR" \
#   -I "$DYNAMATIC/include" \
#   -Wno-deprecated \
#   -o "$SIM_DIR/output_gen"

# Run the C simulation binary
"$SIM_DIR/output_gen"
