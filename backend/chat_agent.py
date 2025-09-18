import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
from datetime import datetime
import hashlib

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

from livekit import rtc, agents
from livekit.agents import JobContext, WorkerOptions, cli

import google.generativeai as genai
from mem0 import Memory
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeRAGMemoryManager:
    """RAG Memory Manager using Pinecone vector database"""

    def __init__(self):
        # Set up OpenAI key (required by mem0 interface but we'll use HuggingFace)
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'dummy-key-for-mem0')
        
        # Try Pinecone with mem0 first
        try:
            self._init_mem0_pinecone()
        except Exception as e:
            logger.warning(f"Failed to initialize mem0 with Pinecone: {e}")
            # Fallback to direct Pinecone integration
            self._init_direct_pinecone()

    def _init_mem0_pinecone(self):
        """Initialize mem0 with Pinecone backend"""
        config = {
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            },
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "api_key": os.environ["PINECONE_API_KEY"],
                    "index_name": "chat-memories",
                    "environment": "us-east-1-aws",  # Pinecone free tier region
                    "embedding_model_dims": 384,
                },
            },
        }

        self.memory = Memory.from_config(config)
        self.use_mem0 = True
        self.conversations = {}  # Fallback storage
        logger.info("✅ mem0 initialized with Pinecone vector database")

    def _init_direct_pinecone(self):
        """Initialize direct Pinecone integration as fallback"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            self.index_name = "chat-memories"
            
            # Create index if it doesn't exist
            existing_indexes = [index['name'] for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # HuggingFace embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Free tier region
                    )
                )
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Initialize HuggingFace embeddings
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            self.memory = None
            self.use_mem0 = False
            self.conversations = {}  # Local fallback
            logger.info("✅ Direct Pinecone integration initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize direct Pinecone: {e}")
            # Ultimate fallback to local storage
            self.memory = None
            self.use_mem0 = False
            self.conversations = {}
            logger.info("🔄 Using local memory as fallback")

    def _generate_id(self, username: str, timestamp: str) -> str:
        """Generate unique ID for vector storage"""
        unique_string = f"{username}-{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    async def add_memory(self, username: str, message: str, response: str):
        """Add conversation to memory system"""
        try:
            timestamp = datetime.now().isoformat()
            
            if self.use_mem0 and self.memory:
                # Use mem0 with Pinecone
                result = await asyncio.to_thread(
                    self.memory.add,
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": response},
                    ],
                    user_id=username,
                    metadata={"timestamp": timestamp},
                )
                logger.info(f"✅ Stored in mem0+Pinecone for {username}")
                return result
                
            elif hasattr(self, 'index'):
                # Direct Pinecone integration
                conversation_text = f"User {username}: {message}\nAI Response: {response}"
                
                # Generate embedding
                embedding = await asyncio.to_thread(
                    self.embedding_model.encode,
                    conversation_text
                )
                
                # Create vector with metadata
                vector_id = self._generate_id(username, timestamp)
                vector_data = {
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": {
                        "username": username,
                        "user_message": message,
                        "ai_response": response,
                        "timestamp": timestamp,
                        "conversation_text": conversation_text
                    }
                }
                
                # Upsert to Pinecone
                await asyncio.to_thread(
                    self.index.upsert,
                    vectors=[vector_data]
                )
                
                logger.info(f"✅ Stored directly in Pinecone for {username}")
                return True
                
            else:
                # Local fallback
                if username not in self.conversations:
                    self.conversations[username] = []
                
                self.conversations[username].append({
                    "user_message": message,
                    "ai_response": response,
                    "timestamp": timestamp
                })
                
                # Keep only last 15 conversations
                if len(self.conversations[username]) > 15:
                    self.conversations[username] = self.conversations[username][-15:]
                
                logger.info(f"✅ Stored in local fallback for {username}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error storing memory for {username}: {e}")
            return False

    async def search_memory(self, username: str, query: str, limit: int = 5):
        """Search for relevant memories"""
        try:
            if self.use_mem0 and self.memory:
                # Use mem0 search
                results = await asyncio.to_thread(
                    self.memory.search,
                    query,
                    username,
                    limit,
                )
                
                formatted_results = []
                for result in results:
                    if isinstance(result, dict):
                        formatted_results.append({
                            "content": result.get("memory", ""),
                            "score": result.get("score", 0.0),
                            "source": "mem0_pinecone"
                        })
                
                logger.info(f"✅ mem0+Pinecone search returned {len(formatted_results)} results for {username}")
                return formatted_results
                
            elif hasattr(self, 'index'):
                # Direct Pinecone search
                query_embedding = await asyncio.to_thread(
                    self.embedding_model.encode,
                    query
                )
                
                # Search Pinecone with user filter
                search_results = await asyncio.to_thread(
                    self.index.query,
                    vector=query_embedding.tolist(),
                    top_k=limit,
                    filter={"username": username},
                    include_metadata=True
                )
                
                formatted_results = []
                for match in search_results.matches:
                    metadata = match.metadata
                    content = f"User: {metadata.get('user_message', '')}\nAI: {metadata.get('ai_response', '')}"
                    formatted_results.append({
                        "content": content,
                        "score": match.score,
                        "source": "direct_pinecone",
                        "timestamp": metadata.get('timestamp', '')
                    })
                
                logger.info(f"✅ Direct Pinecone search returned {len(formatted_results)} results for {username}")
                return formatted_results
                
            else:
                # Local search fallback
                if username not in self.conversations:
                    return []
                
                # Simple keyword matching for fallback
                query_words = query.lower().split()
                scored_results = []
                
                for conv in self.conversations[username]:
                    text = f"{conv['user_message']} {conv['ai_response']}".lower()
                    score = sum(1 for word in query_words if word in text) / len(query_words)
                    
                    if score > 0:
                        scored_results.append({
                            "content": f"User: {conv['user_message']}\nAI: {conv['ai_response']}",
                            "score": score,
                            "source": "local_fallback",
                            "timestamp": conv['timestamp']
                        })
                
                # Sort by score and return top results
                scored_results.sort(key=lambda x: x['score'], reverse=True)
                logger.info(f"✅ Local fallback search returned {len(scored_results[:limit])} results for {username}")
                return scored_results[:limit]
                
        except Exception as e:
            logger.error(f"❌ Error searching memory for {username}: {e}")
            return []

    async def get_user_stats(self, username: str):
        """Get user statistics"""
        try:
            if hasattr(self, 'index'):
                # Query Pinecone for user's total memories
                stats_query = await asyncio.to_thread(
                    self.index.query,
                    vector=[0.0] * 384,  # Dummy vector
                    top_k=1000,  # Large number to get count
                    filter={"username": username},
                    include_metadata=False
                )
                return {"total_memories": len(stats_query.matches)}
            else:
                return {"total_memories": len(self.conversations.get(username, []))}
        except Exception as e:
            logger.error(f"❌ Error getting user stats: {e}")
            return {"total_memories": 0}


class GeminiLLM:
    """Enhanced Gemini LLM for RAG responses"""

    def __init__(self):
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            logger.info("✅ Gemini LLM initialized for RAG")
        except Exception as e:
            logger.error(f"❌ Error initializing Gemini: {e}")
            raise

    async def generate_rag_response(self, message: str, retrieved_contexts: List[Dict], username: str = "User") -> str:
        """Generate response using RAG-retrieved contexts"""
        try:
            if retrieved_contexts:
                # Build context from retrieved memories
                context_parts = []
                for i, ctx in enumerate(retrieved_contexts):
                    score = ctx.get('score', 0)
                    content = ctx.get('content', '')
                    source = ctx.get('source', 'unknown')
                    
                    context_parts.append(f"[Memory {i+1} - {source} - Relevance: {score:.2f}]\n{content}")
                
                context_text = "\n\n".join(context_parts)
                
                prompt = f"""
You are an AI assistant with access to conversation history. Use the retrieved context to provide personalized, accurate responses.

RETRIEVED CONVERSATION HISTORY:
{context_text}

CURRENT USER: {username}
CURRENT MESSAGE: {message}

Instructions:
- Use the conversation history to provide contextual, personalized responses
- Reference specific details from past conversations when relevant
- If the user asks about something mentioned before, use the retrieved context
- Don't explicitly mention that you're using retrieved context - be natural
- If retrieved context is contradictory, acknowledge this
- Be conversational and show continuity with past interactions

Response:
"""
            else:
                prompt = f"""
You are a helpful AI assistant starting a conversation.

USER: {username}
MESSAGE: {message}

Provide a helpful, friendly response:
"""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            # Log context usage
            if retrieved_contexts:
                sources = [ctx.get('source', 'unknown') for ctx in retrieved_contexts]
                avg_score = sum(ctx.get('score', 0) for ctx in retrieved_contexts) / len(retrieved_contexts)
                logger.info(f"RAG response generated using {len(retrieved_contexts)} contexts (avg score: {avg_score:.2f}) from: {', '.join(set(sources))}")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating RAG response: {e}")
            return f"I apologize, {username}, but I'm having trouble generating a response. Please try again."


class PineconeRAGChatAgent:
    """Main RAG Chat Agent using Pinecone"""

    def __init__(self):
        self.memory_manager = PineconeRAGMemoryManager()
        self.llm = GeminiLLM()
        logger.info("✅ Pinecone RAG Chat Agent initialized")

    async def process_message(self, username: str, message: str) -> str:
        """Process message using Pinecone RAG pipeline"""
        
        logger.info(f"🧠 Processing RAG message from {username}: {message}")

        # Step 1: Retrieve relevant contexts from Pinecone
        retrieved_contexts = await self.memory_manager.search_memory(username, message, limit=5)

        # Step 2: Generate response using retrieved context
        response = await self.llm.generate_rag_response(message, retrieved_contexts, username)

        # Step 3: Store new conversation in Pinecone
        await self.memory_manager.add_memory(username, message, response)

        # Log performance metrics
        context_count = len(retrieved_contexts)
        avg_relevance = sum(ctx.get('score', 0) for ctx in retrieved_contexts) / max(context_count, 1)
        
        logger.info(f"✅ RAG response generated for {username}: {context_count} contexts retrieved, avg relevance: {avg_relevance:.3f}")
        
        return response

    async def get_user_memory_stats(self, username: str):
        """Get user's memory statistics"""
        return await self.memory_manager.get_user_stats(username)


# Global agent instance
pinecone_rag_agent = None


async def entrypoint(ctx: JobContext):
    """LiveKit entrypoint for Pinecone RAG agent"""
    
    global pinecone_rag_agent
    
    if pinecone_rag_agent is None:
        pinecone_rag_agent = PineconeRAGChatAgent()
    
    logger.info(f"🚀 Starting Pinecone RAG Agent for room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    # Welcome message
    await ctx.room.local_participant.publish_data(
        payload="🧠 Pinecone RAG AI Agent is ready! I'll remember our conversations using advanced vector search.",
        reliable=True,
        topic="chat"
    )

    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        asyncio.create_task(handle_pinecone_message(data))

    async def handle_pinecone_message(data: rtc.DataPacket):
        try:
            if data.topic == "chat":
                message_data = json.loads(data.data.decode())
                username = message_data.get("username", "anonymous")
                message = message_data.get("message", "")

                if not message.strip():
                    return

                logger.info(f"📨 Received message from {username}: {message}")

                # Process with Pinecone RAG agent
                response = await pinecone_rag_agent.process_message(username, message)

                # Send response
                response_data = {
                    "username": "Pinecone RAG AI",
                    "message": response,
                    "timestamp": datetime.now().isoformat()
                }

                await ctx.room.local_participant.publish_data(
                    payload=json.dumps(response_data),
                    reliable=True,
                    topic="chat"
                )

                logger.info(f"📤 Sent RAG response: {response[:100]}...")

        except Exception as e:
            logger.error(f"❌ Error processing Pinecone message: {e}")
            # Send error response
            error_response = {
                "username": "Pinecone RAG AI",
                "message": "I apologize, but I encountered an error. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

            try:
                await ctx.room.local_participant.publish_data(
                    payload=json.dumps(error_response),
                    reliable=True,
                    topic="chat"
                )
            except:
                pass

    logger.info("🧠 Pinecone RAG Agent is ready and listening...")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )
    )