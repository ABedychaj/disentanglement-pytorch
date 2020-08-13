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
--encoder=ShallowLinear \
--decoder=ShallowLinear \
--max_iter 100000 \
--max_epoch 1000 \
--z_dim=10 \
--batch_size=32 \
--use_wandb=false \
--evaluation_metric mig unsupervised \





