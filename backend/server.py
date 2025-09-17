from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime

from room_manager import RoomManager

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create the main app
app = FastAPI(title="LiveKit AI Chat Agent API")

# Initialize RoomManager as None; will be set during startup
room_manager = None

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class RoomRequest(BaseModel):
    room_name: str

class JoinRoomRequest(BaseModel):
    room_name: str
    username: str

class RoomResponse(BaseModel):
    room_name: str
    access_token: str
    livekit_url: str

# Basic routes
@api_router.get("/")
async def root():
    return {"message": "LiveKit AI Chat Agent API"}

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
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "qdrant_storage": "/tmp/qdrant_chat"
    }

# Startup event to initialize RoomManager
@app.on_event("startup")
async def startup_event():
    global room_manager
    room_manager = RoomManager()
    logger.info("RoomManager initialized during startup")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


# from fastapi import FastAPI, APIRouter, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import os
# import logging
# from pathlib import Path
# from pydantic import BaseModel
# from datetime import datetime

# from room_manager import RoomManager

# ROOT_DIR = Path(__file__).parent
# load_dotenv(ROOT_DIR / '.env')

# # Initialize Room Manager
# room_manager = RoomManager()

# # Create the main app
# app = FastAPI(title="LiveKit AI Chat Agent API")

# # Create a router with the /api prefix
# api_router = APIRouter(prefix="/api")

# # Define Models
# class RoomRequest(BaseModel):
#     room_name: str

# class JoinRoomRequest(BaseModel):
#     room_name: str
#     username: str

# class RoomResponse(BaseModel):
#     room_name: str
#     access_token: str
#     livekit_url: str

# # Basic routes
# @api_router.get("/")
# async def root():
#     return {"message": "LiveKit AI Chat Agent API"}

# # LiveKit Chat Room API endpoints
# @api_router.post("/rooms/create")
# async def create_room(request: RoomRequest):
#     """Create a new chat room"""
#     try:
#         room_info = await room_manager.create_room(request.room_name)
#         return {
#             "success": True,
#             "message": f"Room '{request.room_name}' created successfully",
#             "room_info": room_info
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @api_router.get("/rooms/{room_name}")
# async def get_room_info(room_name: str):
#     """Get information about a specific room"""
#     try:
#         room_info = await room_manager.get_room_info(room_name)
#         if room_info:
#             return {"success": True, "room_info": room_info}
#         else:
#             return {"success": False, "message": "Room not found"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @api_router.post("/rooms/join", response_model=RoomResponse)
# async def join_room(request: JoinRoomRequest):
#     """Generate access token for joining a room"""
#     try:
#         # Check if room exists, create if it doesn't
#         room_info = await room_manager.get_room_info(request.room_name)
#         if not room_info:
#             await room_manager.create_room(request.room_name)
        
#         # Generate access token
#         access_token = room_manager.generate_access_token(
#             request.room_name, 
#             request.username
#         )
        
#         return RoomResponse(
#             room_name=request.room_name,
#             access_token=access_token,
#             livekit_url=os.environ["LIVEKIT_URL"]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @api_router.get("/rooms/{room_name}/participants")
# async def get_room_participants(room_name: str):
#     """Get list of participants in a room"""
#     try:
#         participants = await room_manager.list_participants(room_name)
#         return {
#             "success": True,
#             "room_name": room_name,
#             "participants": participants
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @api_router.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat(),
#         "livekit_configured": bool(os.environ.get("LIVEKIT_URL")),
#         "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
#         "qdrant_storage": "/tmp/qdrant_chat"
#     }

# # Include the router in the main app
# app.include_router(api_router)

# app.add_middleware(
#     CORSMiddleware,
#     allow_credentials=True,
#     allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)