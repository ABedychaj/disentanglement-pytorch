#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=AE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--traverse_z=true \
--encoder=SimpleConv64 \
--decoder=SimpleConv64 \
--max_iter 50000 \
--max_epoch 500 \
--z_dim=32 \
--batch_size=32 \
--use_wandb=false \
--evaluation_metric max_corr \





