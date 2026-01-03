@echo off
echo 🧹 CLEARING PYTHON CACHE...
echo ========================================
echo.

REM Clear all __pycache__ directories recursively
for /r %%i in (__pycache__) do (
    if exist "%%i" (
        echo 🗑️  Removed: %%i
        rd /s /q "%%i"
    )
)

echo.
echo ✅ Cache clearing complete!
echo Run 'clear_cache.bat' anytime to clear cache!
echo.
pause
