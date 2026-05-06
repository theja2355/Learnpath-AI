from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import os, uvicorn

from database import (
    init_db, get_db_connection, rank_items,
    get_user_by_email, create_user
)

SECRET_KEY  = os.environ.get("SECRET_KEY", "smartmoocs-secret-key-2026")
ALGORITHM   = "HS256"
TOKEN_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security    = HTTPBearer(auto_error=False)

app = FastAPI(title="LearnPath AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────
def hash_password(pw):       return pwd_context.hash(pw)
def verify_password(p, h):   return pwd_context.verify(p, h)

def create_token(data):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── Schemas ───────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RecommendRequest(BaseModel):
    skills: str
    role: str

# ── Startup ───────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()

# ── Serve Frontend ────────────────────────────
# Reads index.html from disk — edit the file freely without touching backend.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ── Auth ──────────────────────────────────────
@app.post("/api/register")
def register(req: RegisterRequest):
    if not req.name or not req.email or not req.password:
        raise HTTPException(status_code=400, detail="All fields required")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = create_user(req.name, req.email, hash_password(req.password))
    if not user:
        raise HTTPException(status_code=500, detail="Registration failed")
    token = create_token({"sub": str(user["id"]), "email": user["email"], "name": user["name"]})
    return {"status": "success", "token": token, "name": user["name"], "email": user["email"]}

@app.post("/api/login")
def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token({"sub": str(user["id"]), "email": user["email"], "name": user["name"]})
    return {"status": "success", "token": token, "name": user["name"], "email": user["email"]}

@app.get("/api/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {"name": current_user["name"], "email": current_user["email"]}

# ── Recommend ─────────────────────────────────
@app.post("/api/recommend")
def recommend(req: RecommendRequest, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    courses     = [dict(r) for r in conn.execute("SELECT * FROM courses").fetchall()]
    internships = [dict(r) for r in conn.execute("SELECT * FROM internships").fetchall()]
    conn.close()
    return {
        "status": "success",
        "courses":     rank_items(req.skills, req.role, courses, "course")[:10],
        "internships": rank_items(req.skills, req.role, internships, "internship")[:10],
    }

# ── Resume Upload ─────────────────────────────
@app.post("/api/upload-resume")
async def upload_resume(resume: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    # TODO: integrate a real parser (e.g. pdfminer, docx2txt) to extract skills from the file
    return {
        "status": "success",
        "message": f"Resume '{resume.filename}' uploaded.",
        "extracted_skills": "Python, Machine Learning, Data Analytics",
        "inferred_role": "Data Scientist"
    }

# ── Run ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)
