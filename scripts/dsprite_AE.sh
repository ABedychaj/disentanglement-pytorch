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
--encoder=DeepLinear \
--decoder=DeepLinear \
--max_iter 5000 \
--max_epoch 20000 \
--z_dim=32 \
--batch_size=64 \
--use_wandb=false \





