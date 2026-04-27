from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import uvicorn

from database import (
    init_db, get_db_connection, rank_items,
    get_user_by_email, create_user
)

# ── Config ────────────────────────────────────
SECRET_KEY  = "smartmoocs-secret-key-2026-change-in-production"
ALGORITHM   = "HS256"
TOKEN_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security    = HTTPBearer(auto_error=False)

app = FastAPI(title="Smart MOOCs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────
def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
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

# ── Auth Routes ───────────────────────────────
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

# ── Recommend Route ───────────────────────────
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
    return {
        "status": "success",
        "message": f"Resume '{resume.filename}' uploaded.",
        "extracted_skills": "Python, Machine Learning, Data Analytics",
        "inferred_role": "Data Scientist"
    }

# ── Run ───────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
