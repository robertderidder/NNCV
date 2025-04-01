wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 5 \
    --lr 0.001 \
    --decay 0.7 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab" \