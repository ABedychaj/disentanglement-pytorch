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
--encoder=SimpleConv64 \
--decoder=SimpleConv64 \
--max_iter 200000 \
--max_epoch 1000 \
--z_dim=8 \
--batch_size=64 \
--use_wandb=false \
--number_of_gausses=64 \
--wica_loss=true \
--lambda_wica=10000 \
--recon_lambda=1 \
--print_iter=1000 \
--evaluate_iter=10 \
--evaluation_metric mig factor_vae_metric \
--lr_scheduler=StepLR \
--lr_scheduler_args step_size=1000 gamma=0.9 \



