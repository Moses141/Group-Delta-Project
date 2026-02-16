@echo off
REM Daily model retraining - run at 2am via Windows Task Scheduler
REM Uses data from last 365 days (CSV + database purchases)

cd /d "%~dp0"

echo [%date% %time%] Starting daily model retraining...
python train_models.py
echo [%date% %time%] Retraining complete.
