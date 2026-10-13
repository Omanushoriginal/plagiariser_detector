#!/usr/bin/env bash
echo "========================================================"
echo "  PlagiGuard YT - YouTube Plagiarism Detector Launcher"
echo "========================================================"
echo ""
echo "Starting web server on http://localhost:8000 ..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
