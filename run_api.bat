@echo off
echo Starting Conversational Drift Analyzer REST API Microservice...
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
