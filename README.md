
# 🤖 AI Chat Agent with LiveKit API and Memory-Enhanced Contextual Conversations 

A real-time AI chat app that **remembers conversations** using a vector database.

---

## 🚀 Features
- LiveKit for real-time chat
- Pinecone for memory (vector embeddings)
- Gemini AI as the LLM
- React + FastAPI full-stack app

---

## 🛠️ Setup

### 1. Clone & Install
```bash
git clone https://github.com/KhushamBansal/AI-Chat-Agent
cd AI-Chat-Agent

# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend/vite-project
npm install
````

### 2. Get Free API Keys

* [LiveKit](https://cloud.livekit.io)
* [Pinecone](https://www.pinecone.io)
* [Gemini](https://makersuite.google.com/app/apikey)
* [MongoDB Atlas](https://www.mongodb.com/atlas)

### 3. Configure `.env`

Create `backend/.env`:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-key
LIVEKIT_API_SECRET=your-secret
GEMINI_API_KEY=your-gemini-key
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-OPENAI-key
CORS_ORIGINS="*"
```

---

## ▶️ Run

```bash
# Terminal 1
cd backend && python server.py

# Terminal 2
cd backend && python start_agent.py

# Terminal 3
cd frontend/vite-project && npm run dev
```

Open: 👉 [http://localhost:5173](http://localhost:5173)

---

## 💬 Usage

1. Enter **username** and **room name**
2. Click **Join Room**
3. Chat with the AI — it remembers your past conversations

---

## 📂 Project Structure

```
ai-chat-agent/
│── backend/         # FastAPI + AI agent
│   ├── server.py
│   ├── start_agent.py
│   └── requirements.txt
│
│── frontend/
│   └── vite-project/   # React + Vite app
│
└── README.md
```

---

Built with ❤️ using **React, FastAPI, Pinecone, Gemini AI, LiveKit**

```
