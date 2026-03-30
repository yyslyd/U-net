@echo off
echo ============================================================
echo  UNet Training - AMtown02 15-class (HIGH SCORE)
echo  256x256, BS=8, AMP, AdamW, lr=5e-5
echo  Focal Loss + Data Augmentation + CosineAnnealing
echo  100 epochs, ~2.5 hours on RTX 4060
echo ============================================================
echo.

cd /d D:\assignment2\Pytorch-UNet

set WANDB_MODE=disabled

python -u train.py ^
    --epochs 100 ^
    --batch-size 8 ^
    --learning-rate 5e-5 ^
    --scale 1.0 ^
    --validation 15.0 ^
    --classes 15 ^
    --amp ^
    2>&1 | tee training_full.log

echo.
echo ============================================================
echo  Training complete! Check training_full.log for results.
echo  Best checkpoint in checkpoints/
echo ============================================================
pause
