@echo off
title Adverse Weather ALPR System - Group 6
color 0B

echo ===============================================================================
echo   AUTOMATED LICENSE PLATE CHARACTER RECOGNITION UNDER RAIN, FOG, AND GLARE
echo   Computer Vision Project - Group 6
echo   Members: Aditya Verma (12408804), Varun Singh Rajpurohit (12409154), Appan (12409471)
echo ===============================================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH! Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [*] Starting FastAPI Uvicorn Server...
echo [*] Web Dashboard will be available at: http://localhost:8000
echo.

:: Automatically open browser after 2 seconds in parallel
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

:: Start Uvicorn Web Server
python -m uvicorn app:app --host 127.0.0.1 --port 8000

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server terminated with an error.
    pause
)

