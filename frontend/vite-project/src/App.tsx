import React, { useState, useEffect, useRef } from 'react';
import { Room } from 'livekit-client';
import '@livekit/components-styles';
import './App.css';
import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
const API = `${BACKEND_URL}/api`;

// Define the Message interface
interface Message {
  id: number;
  username: string;
  message: string;
  timestamp: string;
  isBot?: boolean;
  isOwn?: boolean;
  isSystem?: boolean;
}

function App() {
  const [room, setRoom] = useState<Room | null>(null);
  const [username, setUsername] = useState('');
  const [roomName, setRoomName] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (!BACKEND_URL) {
      console.warn('VITE_BACKEND_URL is not defined. Using default: http://localhost:8001');
    }
  }, []);

  const joinRoom = async () => {
    if (!username.trim() || !roomName.trim()) {
      alert('Please enter both username and room name');
      return;
    }

    setIsLoading(true);
    try {
      // Get access token from backend
      const response = await axios.post(`${API}/rooms/join`, {
        room_name: roomName,
        username: username
      });

      const { access_token, livekit_url } = response.data;

      // Create and connect to room
      const room = new Room();
      setRoom(room);

      await room.connect(livekit_url, access_token);
      setIsConnected(true);

      // Listen for data messages (chat)
      room.on('dataReceived', (payload, _participant, _kind, topic) => {
        if (topic === 'chat') {
          try {
            const messageData = JSON.parse(new TextDecoder().decode(payload));
            setMessages(prev => [...prev, {
              id: Date.now() + Math.random(),
              username: messageData.username,
              message: messageData.message,
              timestamp: messageData.timestamp || new Date().toISOString(),
              isBot: messageData.username === 'AI Agent'
            }]);
          } catch (e) {
            console.error('Error parsing message:', e);
          }
        }
      });

      // Add welcome message
      setMessages([{
        id: Date.now(),
        username: 'System',
        message: `Welcome ${username}! You've joined room "${roomName}". The AI agent will respond to your messages.`,
        timestamp: new Date().toISOString(),
        isSystem: true
      }]);

    } catch (error) {
      console.error('Error joining room:', error);
      alert(`Failed to join room: ${error instanceof Error ? error.message : 'Please try again.'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!currentMessage.trim() || !room) return;

    const messageData = {
      username: username,
      message: currentMessage,
      timestamp: new Date().toISOString()
    };

    try {
      // Add user message to local state
      setMessages(prev => [...prev, {
        id: Date.now(),
        username: username,
        message: currentMessage,
        timestamp: new Date().toISOString(),
        isOwn: true
      }]);

      // Send message to room
      const encoder = new TextEncoder();
      const data = encoder.encode(JSON.stringify(messageData));
      await room.localParticipant.publishData(data, { reliable: true, topic: 'chat' });

      setCurrentMessage('');
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      if (isNaN(date.getTime())) {
        return 'Invalid time';
      }
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      console.error('Error formatting timestamp:', e);
      return 'Invalid time';
    }
  };

  const disconnectRoom = () => {
    if (room) {
      room.disconnect();
      room.removeAllListeners();
      setRoom(null);
    }
    setIsConnected(false);
    setMessages([]);
  };

  if (!isConnected) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">AI Chat Agent</h1>
            <p className="text-gray-600">Join a room to start chatting with the AI</p>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Your Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                disabled={isLoading}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Room Name
              </label>
              <input
                type="text"
                value={roomName}
                onChange={(e) => setRoomName(e.target.value)}
                placeholder="Enter room name"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                disabled={isLoading}
              />
            </div>
            
            <button
              onClick={joinRoom}
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Joining...
                </>
              ) : (
                'Join Room'
              )}
            </button>
          </div>
          
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-2">How it works:</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Enter your username and room name</li>
              <li>• Join the room to start chatting</li>
              <li>• The AI agent will remember your conversations</li>
              <li>• Type messages and get intelligent responses</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-800">AI Chat Agent</h1>
            <p className="text-sm text-gray-600">Room: {roomName} | User: {username}</p>
          </div>
          <button
            onClick={disconnectRoom}
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Leave Room
          </button>
        </div>
      </div>

      {/* Chat Container */}
      <div className="flex flex-col h-[calc(100vh-80px)]">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.isOwn ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-md px-4 py-3 rounded-2xl ${
                    msg.isSystem
                      ? 'bg-yellow-100 border border-yellow-200 text-yellow-800'
                      : msg.isBot
                      ? 'bg-indigo-500 text-white'
                      : msg.isOwn
                      ? 'bg-blue-500 text-white'
                      : 'bg-white border border-gray-200 text-gray-800'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-xs font-medium ${
                      msg.isSystem ? 'text-yellow-700' : 
                      msg.isBot ? 'text-indigo-100' : 
                      msg.isOwn ? 'text-blue-100' : 'text-gray-500'
                    }`}>
                      {msg.username}
                    </span>
                    <span className={`text-xs ${
                      msg.isSystem ? 'text-yellow-600' : 
                      msg.isBot ? 'text-indigo-200' : 
                      msg.isOwn ? 'text-blue-200' : 'text-gray-400'
                    }`}>
                      {formatTime(msg.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {msg.message}
                  </p>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Message Input */}
        <div className="bg-white border-t px-6 py-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex space-x-4">
              <textarea
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your message... (Press Enter to send)"
                className="flex-1 px-4 py-3 border border-gray-300 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none resize-none"
                rows={1}
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
              <button
                onClick={sendMessage}
                disabled={!currentMessage.trim()}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-6 py-3 rounded-2xl transition-colors font-medium"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;