from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime

from room_manager import RoomManager

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize Room Manager
room_manager = RoomManager()

# Create the main app without a prefix
app = FastAPI(title="LiveKit AI Chat Agent API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class RoomRequest(BaseModel):
    room_name: str

class JoinRoomRequest(BaseModel):
    room_name: str
    username: str

class RoomResponse(BaseModel):
    room_name: str
    access_token: str
    livekit_url: str

class ParticipantInfo(BaseModel):
    identity: str
    name: str
    state: str
    joined_at: int


# Original routes
@api_router.get("/")
async def root():
    return {"message": "LiveKit AI Chat Agent API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]


# LiveKit Chat Room API endpoints
@api_router.post("/rooms/create")
async def create_room(request: RoomRequest):
    """Create a new chat room"""
    try:
        room_info = await room_manager.create_room(request.room_name)
        return {
            "success": True,
            "message": f"Room '{request.room_name}' created successfully",
            "room_info": room_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/rooms/{room_name}")
async def get_room_info(room_name: str):
    """Get information about a specific room"""
    try:
        room_info = await room_manager.get_room_info(room_name)
        if room_info:
            return {"success": True, "room_info": room_info}
        else:
            return {"success": False, "message": "Room not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/rooms/join", response_model=RoomResponse)
async def join_room(request: JoinRoomRequest):
    """Generate access token for joining a room"""
    try:
        # Check if room exists, create if it doesn't
        room_info = await room_manager.get_room_info(request.room_name)
        if not room_info:
            await room_manager.create_room(request.room_name)
        
        # Generate access token
        access_token = room_manager.generate_access_token(
            request.room_name, 
            request.username
        )
        
        return RoomResponse(
            room_name=request.room_name,
            access_token=access_token,
            livekit_url=os.environ["LIVEKIT_URL"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/rooms/{room_name}/participants")
async def get_room_participants(room_name: str):
    """Get list of participants in a room"""
    try:
        participants = await room_manager.list_participants(room_name)
        return {
            "success": True,
            "room_name": room_name,
            "participants": participants
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "livekit_configured": bool(os.environ.get("LIVEKIT_URL")),
        "mem0_configured": bool(os.environ.get("MEM0_API_KEY")),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY"))
    }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def init_room_manager():
    await room_manager.init()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
