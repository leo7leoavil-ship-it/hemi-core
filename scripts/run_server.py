"""Development and production server runner."""

import os
import sys
import uvicorn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_settings


def main():
    settings = get_settings()
    print(f"Starting {settings.PROJECT_NAME} on {settings.HOST}:{settings.PORT}...")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
    )


if __name__ == "__main__":
    main()
