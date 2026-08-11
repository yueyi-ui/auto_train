@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHON=E:\anaconda\envs\xanylabel\python.exe
"%PYTHON%" run_auto_train.py %*
