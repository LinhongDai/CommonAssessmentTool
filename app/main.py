"""
Main application module for the Common Assessment Tool.
This module initializes the FastAPI application and includes all routers.
Handles database initialization and CORS middleware configuration.
"""

from fastapi import FastAPI
from app.model.router import router as model_router
from app import models
from app.database import engine
from app.clients.router import router as clients_router
from app.auth.router import router as auth_router
from fastapi.middleware.cors import CORSMiddleware
from initialize_data import initialize_database


# Initialize database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI application
app = FastAPI(
    title="Case Management API",
    description="API for managing client cases",
    version="1.0.0",
)

# Auto-initialize database when app starts
@app.on_event("startup")
async def startup_event():
    initialize_database()


# Include routers
app.include_router(auth_router)
app.include_router(clients_router)
app.include_router(model_router)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    allow_credentials=True,
)
