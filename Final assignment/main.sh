wandb login

python3 overtrain.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 200\
    --lr1 0.001 \
    --lr2 0.001 \
    --decay 0.5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dice_overfit" \