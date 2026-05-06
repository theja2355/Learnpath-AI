# LearnPath AI — Setup & Run Guide

## Project Structure

```
learnpath_ai/
├── backend.py        ← FastAPI server + all API routes
├── database.py       ← SQLite setup + TF-IDF recommendation engine
├── index.html        ← Frontend (served by backend at /)
├── requirements.txt  ← Python dependencies
└── moocs.db          ← Auto-created on first run
```

## How It's Connected

```
Browser → GET /          → backend serves index.html
Browser → POST /api/...  → FastAPI handles auth & recommendations
FastAPI → database.py    → SQLite (moocs.db)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python backend.py
```
or
```bash
uvicorn backend:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Open in browser
```
http://localhost:8080
```

That's it! The database (`moocs.db`) is created automatically on first run
and seeded with 60 courses and 37 internships.

## API Endpoints

| Method | Endpoint              | Auth | Description              |
|--------|-----------------------|------|--------------------------|
| GET    | /                     | No   | Serve frontend           |
| POST   | /api/register         | No   | Register new user        |
| POST   | /api/login            | No   | Login, returns JWT token |
| GET    | /api/me               | JWT  | Get current user info    |
| POST   | /api/recommend        | JWT  | Get course + intern recs |
| POST   | /api/upload-resume    | JWT  | Upload resume file       |

## Environment Variables (optional)

| Variable    | Default                       | Description          |
|-------------|-------------------------------|----------------------|
| SECRET_KEY  | smartmoocs-secret-key-2026    | JWT signing secret   |
| PORT        | 8080                          | Server port          |

Set them before running:
```bash
export SECRET_KEY="your-strong-secret"
export PORT=8000
python backend.py
```
