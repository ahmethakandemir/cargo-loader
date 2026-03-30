@echo off
title CargoLoader
echo Starting CargoLoader...
python -m cargoloader
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error running CargoLoader.
    echo Make sure Python and dependencies are installed:
    echo   pip install -r requirements.txt
)
pause
