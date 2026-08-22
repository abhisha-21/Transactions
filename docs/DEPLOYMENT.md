# Deployment Guide

## Local Development

```bash
pip install -r deployment/requirements.txt
python src/api.py
```

API runs at: http://localhost:5000

## Docker

```bash
docker build -t ai-risk-manager:latest -f deployment/Dockerfile .
docker run -p 5000:5000 ai-risk-manager:latest
```

## Production Checklist

- [ ] API authentication enabled
- [ ] HTTPS/TLS enabled
- [ ] Logging active
- [ ] Monitoring setup