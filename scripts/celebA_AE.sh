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
--traverse_z=true \
--encoder=SimpleConv64 \
--decoder=SimpleConv64 \
--batch_size 128 \
--max_iter 100000 \
--max_epoch 1000 \
--z_dim=6 \
--batch_size=32 \
--use_wandb=false \
--evaluation_metric unsupervised \
