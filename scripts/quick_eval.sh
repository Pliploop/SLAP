#!/bin/bash

CONFIG_FILE=config/clap.yaml

# checkpoint to evaluate (made absolute)
ORIG_WEIGHT_FILE="$(cd "$(dirname "$1")" || exit 1; pwd)/$(basename "$1")"

# temporary file to write the results into (useful if the script is called from a subprocess)
TMP_FILE=$2

# create a temporary checkpoint to ensure that the checkpoint is not overriden during evaluation
WEIGHT_FILE=$(mktemp "$ORIG_WEIGHT_FILE".XXXX)

cp "$ORIG_WEIGHT_FILE" "$WEIGHT_FILE" || exit 1

# remove temporary file when the program terminates (similar to a `finally` clause)
trap 'rm $WEIGHT_FILE' EXIT

# device
if [ -n "$3" ]; then
  GPU="$3"
else
  GPU=0
fi

cd evar || exit 1

CUDA_VISIBLE_DEVICES=$GPU python lineareval.py $CONFIG_FILE gtzan batch_size=64,weight_file="$WEIGHT_FILE" "${@:3}"
CUDA_VISIBLE_DEVICES=$GPU python lineareval.py $CONFIG_FILE mtt-tag batch_size=64,weight_file="$WEIGHT_FILE" "${@:3}"

python summarize.py "$WEIGHT_FILE" "$TMP_FILE"
