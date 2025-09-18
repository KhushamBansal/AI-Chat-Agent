#!/usr/bin/env python3


import asyncio
import os
import sys
from pathlib import Path
from chat_agent import entrypoint
try:
    from livekit.agents import WorkerOptions, cli
except ImportError:
    raise ImportError("livekit-agents package is not installed. Please install it to run this agent.")

# Set the working directory to the backend folder
backend_dir = Path(__file__).parent
os.chdir(backend_dir)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
except ImportError:
    raise ImportError("python-dotenv package is not installed. Please install it to load .env files.")

def main():
    """Start the LiveKit agent"""
    try:
        print("Starting LiveKit AI Chat Agent...")
        print(f"LiveKit URL: {os.environ.get('LIVEKIT_URL')}")
        print(f"Working directory: {os.getcwd()}")

        # Create worker options
        worker_opts = WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )

        # Run the agent
        cli.run_app(worker_opts)

    except Exception as e:
        print(f"Error starting agent: {e}")
        return 1

if __name__ == "__main__":
    main()