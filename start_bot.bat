@echo off
REM ========================================
REM Quadrick AI Trading Bot - Windows Starter
REM ========================================

echo ========================================
echo    QUADRICK AI TRADING SYSTEM
echo    Mission: $15 -^> $100,000
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found.
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check if .env exists
if not exist ".env" (
    if exist "config\env.example" (
        echo Creating .env file from template...
        copy config\env.example .env
        echo.
        echo =======================================
        echo  IMPORTANT: Edit .env file with your
        echo  API keys before continuing!
        echo =======================================
        pause
    )
)

REM Display menu
:menu
echo.
echo Select an option:
echo   1. Run Setup Wizard
echo   2. Test Connections
echo   3. Start Trading Bot
echo   4. Start Bot (Background)
echo   5. View Logs
echo   6. Exit
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo Running setup wizard...
    python setup.py
    pause
    goto menu
)

if "%choice%"=="2" (
    echo Testing connections...
    python test_connection.py
    pause
    goto menu
)

if "%choice%"=="3" (
    echo Starting trading bot...
    echo Press Ctrl+C to stop
    echo.
    python main.py
    pause
    goto menu
)

if "%choice%"=="4" (
    echo Starting bot in background...
    start /min python main.py
    echo Bot started in background window
    pause
    goto menu
)

if "%choice%"=="5" (
    if exist "logs\trading.log" (
        echo Opening log file...
        start notepad logs\trading.log
    ) else (
        echo No log file found yet.
    )
    pause
    goto menu
)

if "%choice%"=="6" (
    echo Goodbye!
    exit
)

echo Invalid choice. Please try again.
goto menu
