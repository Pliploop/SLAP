#!/bin/bash

CONFIG_FILE=config/clap.yaml

# checkpoint to evaluate (made absolute)

#if s3 in the path, download the checkpoint to the local machine with basename as the filename
if [[ $1 == s3://* ]]; then
  echo "Downloading checkpoint from S3"
  
  #make evar/temp/ckpt directory if it doesn't exist
  mkdir -p evar/temp/ckpt
  
  #job name is two parents up from s3 base path
  JOB_NAME=$(basename $(dirname $(dirname $1)))
  
  python <<EOF 
import wandb
import os

api = wandb.Api()
job_name = "$JOB_NAME"
runs = api.runs('SLAP', filters={'config.job_name': job_name})

save_dir = 'evar/temp'  # Define target directory
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

if runs:
    run = runs[0]  # Assuming you want the first matching run
    config_path = run.file('config.yaml').download(root=save_dir, replace=True).name
    ## load the config file
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        ## keep only data and model keys
        config = {k: v['value'] for k, v in config.items() if k in ['data', 'model']}
    ## save the config file
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Config file saved as: {config_path}')
else:
    print(f'No run found with job_name={job_name}')
EOF
  
  aws s3 cp $1 evar/temp/ckpt/$(basename $1)



fi

if [[ $1 == s3://* ]]; then
# make absolute path with basename and pwd without dirname
  ORIG_WEIGHT_FILE="$(pwd)/evar/temp/ckpt/$(basename "$1")"
  CONFIG_FILE_S3="$(pwd)/evar/temp/config.yaml"
else
  ORIG_WEIGHT_FILE="$(cd "$(dirname "$1")" || exit 1; pwd)/$(basename "$1")"
fi

# temporary file to write the results into (useful if the script is called from a subprocess)
TMP_FILE=$2

# create a temporary checkpoint to ensure that the checkpoint is not overriden during evaluation
WEIGHT_FILE=$(mktemp "$ORIG_WEIGHT_FILE".XXXX)

cp "$ORIG_WEIGHT_FILE" "$WEIGHT_FILE" || exit 1

# remove temporary file when the program terminates (similar to a `finally` clause)
trap 'rm $WEIGHT_FILE' EXIT
## also remove the original checkpoint file if it was downloaded from S3
if [[ $1 == s3://* ]]; then
  trap 'rm $CONFIG_FILE_S3; rm $WEIGHT_FILE; rm $ORIG_WEIGHT_FILE' EXIT
  
fi

# device
if [ -n "$3" ]; then
  GPU="$3"
else
  GPU=0
fi

cd evar || exit 1

# CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE gtzan batch_size=64,weight_file="$WEIGHT_FILE" "${@:3}"
# CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE giantsteps-key batch_size=64,weight_file="$WEIGHT_FILE" "${@:3}"
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE mtt-tag batch_size=64,weight_file="$WEIGHT_FILE" "${@:3}"
# CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE openmic-tag batch_size=64,weight_file="$WEIGHT_FILE" "${@:3}"
# CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE nsynth batch_size=64,weight_file="$WEIGHT_FILE" "${@:3}"

python summarize.py "$WEIGHT_FILE" "$TMP_FILE"
