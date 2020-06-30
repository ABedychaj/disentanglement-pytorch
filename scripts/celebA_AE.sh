#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=AE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA \
--dset_name=celebA \
--encoder=ShallowLinear \
--decoder=ShallowLinear \
--batch_size 32 \
--z_dim=128 \
--use_wandb=false \



