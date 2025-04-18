wandb login

python3 trainunet.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100\
    --lr1 0.001 \
    --lr2 0.0003 \
    --decay 0.5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet_final" \