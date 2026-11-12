from app.main import app

try:
    from app.main import demo
except ImportError:
    demo = None

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
