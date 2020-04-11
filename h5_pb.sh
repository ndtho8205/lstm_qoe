#!/usr/bin/env zsh

SCRIPT_DIR="$(
  cd "$(dirname "$0")"
  pwd -P
)"

H5_PATH="${SCRIPT_DIR}/LFOVIA_QoE/final_model.h5"
PB_PATH="${SCRIPT_DIR}/LFOVIA_QoE/mobile_model.pb"

python "${SCRIPT_DIR}/keras_to_tensorflow/keras_to_tensorflow.py" \
  --input_model="$H5_PATH" \
  --output_model="$PB_PATH"
