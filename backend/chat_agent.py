import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

from livekit import rtc, agents
from livekit.agents import JobContext, WorkerOptions, cli

import google.generativeai as genai
from mem0 import Memory
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryManager:
    """Handles memory operations using mem0 RAG system"""
    
    def __init__(self):
        # Configure mem0 with HuggingFace embeddings
        config = {
            'embedder': {
                'provider': 'huggingface',
                'config': {
                    'model': 'multi-qa-MiniLM-L6-cos-v1'
                }
            },
            'vector_store': {
                'provider': 'qdrant',
                'config': {
                    'collection_name': 'chat_memories',
                    'embedding_model_dims': 384,
                    'on_disk': True,
                    'path': '/tmp/qdrant_chat'
                }
            }
        }
        
        try:
            self.memory = Memory.from_config(config)
            logger.info("✅ mem0 Memory initialized successfully")
        except Exception as e:
            logger.error(f"⚠️ Error initializing mem0: {e}")
            self.memory = None
            self.conversations = {}
    
    async def add_memory(self, username: str, message: str, response: str):
        """Add conversation to mem0 vector database"""
        try:
            if self.memory:
                result = self.memory.add(
                    messages=[
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": response}
                    ],
                    user_id=username,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "conversation_text": f"User {username} said: {message}. AI responded: {response}"
                    }
                )
                logger.info(f"✅ Added conversation to mem0 for user {username}")
                return result
            else:
                # Fallback to simple storage
                if username not in self.conversations:
                    self.conversations[username] = []
                
                self.conversations[username].append({
                    "user_message": message,
                    "ai_response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Keep only last 10 conversations per user
                if len(self.conversations[username]) > 10:
                    self.conversations[username] = self.conversations[username][-10:]
                
                logger.info(f"✅ Added conversation to fallback memory for user {username}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error adding memory for {username}: {e}")
            return None
    
    async def search_memory(self, username: str, query: str, limit: int = 5):
        """Search for relevant memories using mem0 vector search"""
        try:
            if self.memory:
                results = self.memory.search(
                    query=query,
                    user_id=username,
                    limit=limit
                )
                
                formatted_results = []
                for result in results:
                    if isinstance(result, dict):
                        memory_text = result.get('memory', '')
                        score = result.get('score', 0)
                        
                        formatted_results.append({
                            "text": memory_text,
                            "relevance_score": score,
                            "timestamp": result.get('created_at', '')
                        })
                
                logger.info(f"📚 Found {len(formatted_results)} relevant memories for user {username}")
                return formatted_results
            else:
                # Fallback to simple search
                if username not in self.conversations:
                    return []
                
                recent_conversations = self.conversations[username][-limit:]
                results = []
                for conv in recent_conversations:
                    results.append({
                        "text": f"User said: {conv['user_message']} | AI responded: {conv['ai_response']}",
                        "timestamp": conv["timestamp"]
                    })
                
                logger.info(f"📝 Found {len(results)} memories using fallback for user {username}")
                return results
                
        except Exception as e:
            logger.error(f"❌ Error searching memory for {username}: {e}")
            return []


class GeminiLLM:
    """Gemini LLM integration with enhanced context handling"""
    
    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def generate_response(self, prompt: str, context: str = "", username: str = "User") -> str:
        """Generate response using Gemini with mem0 context"""
        try:
            if context:
                full_prompt = f"""You are an AI assistant with access to previous conversation history. Use this context to provide personalized, contextual responses.

PREVIOUS CONVERSATION CONTEXT:
{context}

CURRENT USER: {username}
CURRENT MESSAGE: {prompt}

Instructions:
- Reference relevant information from previous conversations when appropriate
- Be conversational and natural
- Show that you remember previous interactions
- If the user asks about past conversations, use the context to answer accurately
- Keep responses helpful and engaging

Please provide a response:"""
            else:
                full_prompt = f"""You are a helpful AI assistant. This appears to be the start of a new conversation.

USER: {username}
MESSAGE: {prompt}

Please provide a helpful, friendly response:"""
            
            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error generating Gemini response: {e}")
            return f"I apologize, {username}, but I'm having trouble generating a response right now. Please try again."


class ChatAgent:
    """Main chat agent class with mem0 RAG integration"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.llm = GeminiLLM()
        
    async def process_message(self, username: str, message: str) -> str:
        """Process incoming message with mem0 RAG and generate response"""
        
        logger.info(f"🤖 Processing message from {username}: {message}")
        
        # Search for relevant memories
        memories = await self.memory_manager.search_memory(username, message, limit=3)
        
        # Build context from relevant memories
        context = ""
        if memories:
            context_parts = []
            for memory in memories:
                if isinstance(memory, dict):
                    memory_text = memory.get('text', '')
                    relevance_score = memory.get('relevance_score', 0)
                    
                    # Only include highly relevant memories
                    if relevance_score > 0.5 or not memory.get('relevance_score'):
                        context_parts.append(memory_text)
            
            if context_parts:
                context = "\n".join(context_parts[:3])
                logger.info(f"📚 Using context for {username}: {len(context_parts)} relevant memories")
            else:
                logger.info(f"🔍 No highly relevant memories found for {username}")
        else:
            logger.info(f"💭 No previous memories found for {username}")
        
        # Generate response using Gemini with context
        response = await self.llm.generate_response(message, context, username)
        
        # Add this conversation to mem0 for future reference
        await self.memory_manager.add_memory(username, message, response)
        
        logger.info(f"✅ Generated response for {username} (length: {len(response)} chars)")
        return response


# Global agent instance
chat_agent = ChatAgent()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent"""
    
    logger.info(f"🚀 Starting agent for room: {ctx.room.name}")
    
    # Connect to the room
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    
    # Send initial message
    await ctx.room.local_participant.publish_data(
        payload="AI Agent has joined the chat! 🤖",
        reliable=True,
        topic="chat"
    )
    
    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        """Handle incoming chat messages"""
        asyncio.create_task(handle_data_message(data))
    
    async def handle_data_message(data: rtc.DataPacket):
        """Handle incoming data messages asynchronously"""
        try:
            if data.topic == "chat":
                message_data = json.loads(data.data.decode())
                username = message_data.get("username", "anonymous")
                message = message_data.get("message", "")
                
                logger.info(f"📨 Received message from {username}: {message}")
                
                # Process message and generate response
                response = await chat_agent.process_message(username, message)
                
                # Send response back to the room
                response_data = {
                    "username": "AI Agent",
                    "message": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                await ctx.room.local_participant.publish_data(
                    payload=json.dumps(response_data),
                    reliable=True,
                    topic="chat"
                )
                
                logger.info(f"📤 Sent response: {response}")
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    # Keep the agent running
    logger.info("🎯 Agent is ready and listening for messages...")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )
    )