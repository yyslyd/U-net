@echo off
setlocal
cd /d "%~dp0"
set WANDB_MODE=disabled
echo Training AMtown02 image segmentation. Activate your Python environment first.
echo Output is saved to training_local.log. Additional arguments override defaults.
python -u train.py --epochs 100 --batch-size 8 --learning-rate 5e-5 --scale 1.0 --validation 15 --classes 15 --seed 0 --no-wandb %* > training_local.log 2>&1
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" (
  echo Training failed. See training_local.log.
) else (
  echo Training finished. See training_local.log and the checkpoint directory.
)
exit /b %TRAIN_EXIT%
