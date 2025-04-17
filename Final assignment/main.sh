wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 150\
    --lr1 0.001 \
    --lr2 0.0003 \
    --decay 0.5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab_final" \