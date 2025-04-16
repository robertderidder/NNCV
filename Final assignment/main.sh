wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 50\
    --lr1 0.001 \
    --lr2 0.0001 \
    --decay 0.5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab_cos_back_32" \