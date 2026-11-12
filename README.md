---
title: MediStock OCR Backend
emoji: 💊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# MediStock PaddleOCR Backend

Production OCR backend for MediStock Android ERP.

## Endpoints:
- `GET /health`: Engine health check
- `POST /ocr/parse-bill`: Full invoice parsing with spatial row clustering and mathematical validation
- `/`: Interactive Gradio web testing interface
