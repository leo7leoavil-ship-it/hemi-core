"""API v1 Router aggregating all endpoint routes."""

from fastapi import APIRouter
from app.api.v1.endpoints import evaluate, health, query, websocket

api_v1_router = APIRouter()

api_v1_router.include_router(health.router)
api_v1_router.include_router(query.router)
api_v1_router.include_router(evaluate.router)
api_v1_router.include_router(websocket.router)
