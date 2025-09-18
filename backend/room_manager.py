import os
import asyncio
from livekit import api
from livekit.api import AccessToken, VideoGrants
import logging
import aiohttp

logger = logging.getLogger(__name__)


class RoomManager:
    """Manages LiveKit rooms and access tokens"""

    def __init__(self):
        self.livekit_api = None

    async def init(self):
        """Initialize LiveKit API client with aiohttp session"""
        session = aiohttp.ClientSession()
        self.livekit_api = api.LiveKitAPI(
            url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
            session=session,
        )

    async def create_room(self, room_name: str) -> dict:
        """Create a new LiveKit room"""
        try:
            room_opts = api.CreateRoomRequest(name=room_name)
            room = await self.livekit_api.room.create_room(room_opts)
            logger.info(f"Created room: {room_name}")
            return {
                "room_name": room.name,
                "creation_time": room.creation_time,
                "sid": room.sid,
            }
        except Exception as e:
            logger.error(f"Error creating room {room_name}: {e}")
            raise e

    async def get_room_info(self, room_name: str) -> dict:
        """Get information about a room"""
        try:
            rooms = await self.livekit_api.room.list_rooms(api.ListRoomsRequest())
            for room in rooms:
                if room.name == room_name:
                    return {
                        "room_name": room.name,
                        "sid": room.sid,
                        "num_participants": room.num_participants,
                        "creation_time": room.creation_time,
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting room info for {room_name}: {e}")
            return None

    def generate_access_token(self, room_name: str, username: str) -> str:
        """Generate access token for a user to join a room"""
        try:
            token = AccessToken(
                api_key=os.environ["LIVEKIT_API_KEY"],
                api_secret=os.environ["LIVEKIT_API_SECRET"],
            )

            # Grant permissions
            grant = VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )

            token.with_grants(grant).with_identity(username)

            access_token = token.to_jwt()
            logger.info(f"Generated access token for {username} in room {room_name}")
            return access_token

        except Exception as e:
            logger.error(f"Error generating access token: {e}")
            raise e

    async def list_participants(self, room_name: str) -> list:
        """List participants in a room"""
        try:
            participants = await self.livekit_api.room.list_participants(
                api.ListParticipantsRequest(room=room_name)
            )
            return [
                {
                    "identity": p.identity,
                    "name": p.name,
                    "state": p.state,
                    "joined_at": p.joined_at,
                }
                for p in participants
            ]
        except Exception as e:
            logger.error(f"Error listing participants for {room_name}: {e}")
            return []
