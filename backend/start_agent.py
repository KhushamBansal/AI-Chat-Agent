#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from chat_agent import entrypoint
from livekit.agents import WorkerOptions, cli

# Set the working directory to the backend folder
backend_dir = Path(__file__).parent
os.chdir(backend_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv('.env')

def main():
    """Start the LiveKit agent"""
    try:
        print("🚀 Starting LiveKit AI Chat Agent...")
        print(f"📡 LiveKit URL: {os.environ.get('LIVEKIT_URL')}")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🤖 Gemini configured: {bool(os.environ.get('GEMINI_API_KEY'))}")
        print(f"💾 Memory storage: /tmp/qdrant_chat")
        
        # Validate required environment variables
        required_vars = ['LIVEKIT_URL', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'GEMINI_API_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            return 1
        
        # Create worker options
        worker_opts = WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )
        
        print("✅ All configurations validated. Starting agent...")
        
        # Run the agent with 'start' subcommand by default
        sys.argv = sys.argv[:1] + ['start'] + sys.argv[1:]
        cli.run_app(worker_opts)
        
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        return 1

if __name__ == "__main__":
    exit(main())