wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 5\
    --lr1 0.01 \
    --lr2 0.0001 \
    --decay 0.5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab" \