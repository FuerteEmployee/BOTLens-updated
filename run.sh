#!/bin/bash
cd backend
source venv/bin/activate
echo "Starting Aether Vision with SSL... (Access via HTTPS)"
uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
