"""
server/app.py
FastAPI server wiring for LPG Inspector environment.

Endpoints created automatically:
    /ws      — WebSocket (primary, used by training loops)
    /reset   — POST HTTP reset
    /step    — POST HTTP step
    /state   — GET HTTP state
    /health  — GET health check (must return 200)
    /web     — Web UI
    /docs    — Swagger API docs
"""

from openenv.core.env_server import create_fastapi_app
from server.environment import LPGInspectorEnvironment
from models import LPGInspectorAction, LPGInspectorObservation

app = create_fastapi_app(LPGInspectorEnvironment, LPGInspectorAction, LPGInspectorObservation)