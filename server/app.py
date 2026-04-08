"""
server/app.py
FastAPI server wiring for LPG Inspector environment.
"""

import uvicorn
from openenv.core.env_server import create_fastapi_app
from server.environment import LPGInspectorEnvironment
from models import LPGInspectorAction, LPGInspectorObservation

app = create_fastapi_app(LPGInspectorEnvironment, LPGInspectorAction, LPGInspectorObservation)


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, workers=4)


if __name__ == "__main__":
    main()