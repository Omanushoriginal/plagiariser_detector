@echo off
echo ========================================================
echo   PlagiGuard YT - YouTube Plagiarism Detector Launcher
echo ========================================================
echo.
echo Starting web server on http://localhost:8000 ...
py -3.13 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
if errorlevel 1 (
    echo Python 3.13 launcher not found, trying python ...
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
)
pause
