import io
import re
from typing import Tuple

from pypdf import PdfReader
from docx import Document

SKILL_VOCAB = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R", "MATLAB", "Dart",
    "HTML", "CSS", "Tailwind", "Bootstrap", "SASS",
    "React", "Next.js", "Vue", "Angular", "Svelte", "Redux",
    "Node.js", "Express", "Flask", "Django", "FastAPI", "Spring", "Spring Boot",
    "REST API", "GraphQL", "gRPC", "Microservices",
    "SQL", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Cassandra",
    "Elasticsearch", "DynamoDB", "Firebase",
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "Terraform",
    "Jenkins", "CI/CD", "GitHub Actions", "Linux", "Bash", "DevOps",
    "Machine Learning", "Deep Learning", "Neural Networks", "TensorFlow",
    "PyTorch", "Keras", "scikit-learn", "Pandas", "NumPy", "SciPy",
    "Matplotlib", "Seaborn", "OpenCV", "NLP", "Computer Vision",
    "Reinforcement Learning", "LLM", "Generative AI", "Transformers", "BERT",
    "Hugging Face", "MLOps",
    "Data Analysis", "Data Analytics", "Data Science", "Data Engineering",
    "ETL", "Spark", "Hadoop", "Kafka", "Airflow", "Tableau", "Power BI", "Excel",
    "iOS", "Android", "React Native", "Flutter", "Xcode",
    "Cybersecurity", "Penetration Testing", "Ethical Hacking", "Cryptography",
    "Blockchain", "Solidity", "Web3", "Ethereum",
    "Agile", "Scrum", "Project Management", "Product Management",
    "UI/UX", "Figma", "Adobe XD", "Photoshop",
    "Git", "GitHub", "GitLab",
    "Statistics", "Probability", "Linear Algebra",
]


def _extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _extract_text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_text_from_txt(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return _extract_text_from_docx(data)
    if name.endswith(".txt"):
        return _extract_text_from_txt(data)
    # Best effort: try PDF, then docx, then txt
    for fn in (_extract_text_from_pdf, _extract_text_from_docx, _extract_text_from_txt):
        try:
            text = fn(data)
            if text.strip():
                return text
        except Exception:
            continue
    return ""


def find_skills(text: str) -> list:
    if not text:
        return []
    found = []
    seen = set()
    lower_text = text.lower()
    for skill in SKILL_VOCAB:
        # Use word-boundary search; allow tokens with special chars (C++, C#, .NET)
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(skill.lower()) + r"(?![A-Za-z0-9_])"
        if re.search(pattern, lower_text):
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                found.append(skill)
    return found


ROLE_RULES = [
    ("Machine Learning Engineer",
     ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "scikit-learn", "Neural Networks", "MLOps"]),
    ("Data Scientist",
     ["Data Science", "Pandas", "NumPy", "Statistics", "scikit-learn", "R", "Machine Learning", "Data Analysis"]),
    ("Data Engineer",
     ["ETL", "Spark", "Hadoop", "Kafka", "Airflow", "Data Engineering", "SQL"]),
    ("Data Analyst",
     ["Data Analysis", "Data Analytics", "SQL", "Excel", "Tableau", "Power BI", "Pandas"]),
    ("AI / NLP Engineer",
     ["NLP", "LLM", "Transformers", "BERT", "Hugging Face", "Generative AI"]),
    ("Frontend Developer",
     ["React", "Vue", "Angular", "Next.js", "TypeScript", "JavaScript", "HTML", "CSS", "Tailwind"]),
    ("Backend Developer",
     ["Node.js", "Express", "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "REST API", "GraphQL"]),
    ("Full Stack Developer",
     ["React", "Node.js", "MongoDB", "PostgreSQL", "Express", "Next.js"]),
    ("Mobile Developer",
     ["iOS", "Android", "Swift", "Kotlin", "React Native", "Flutter", "Dart"]),
    ("DevOps Engineer",
     ["Docker", "Kubernetes", "Terraform", "Jenkins", "CI/CD", "AWS", "Azure", "GCP", "DevOps"]),
    ("Cloud Engineer",
     ["AWS", "Azure", "GCP", "Google Cloud", "Terraform", "Kubernetes"]),
    ("Cybersecurity Analyst",
     ["Cybersecurity", "Penetration Testing", "Ethical Hacking", "Cryptography"]),
    ("Blockchain Developer",
     ["Blockchain", "Solidity", "Ethereum", "Web3"]),
    ("UI/UX Designer",
     ["UI/UX", "Figma", "Adobe XD", "Photoshop"]),
]


def infer_role(skills: list) -> str:
    if not skills:
        return "Software Developer"
    skill_set = {s.lower() for s in skills}
    best_role = "Software Developer"
    best_score = 0
    for role, keywords in ROLE_RULES:
        score = sum(1 for k in keywords if k.lower() in skill_set)
        if score > best_score:
            best_score = score
            best_role = role
    if best_score == 0:
        return "Software Developer"
    return best_role


def parse_resume(filename: str, data: bytes) -> Tuple[str, str]:
    text = extract_text(filename, data)
    skills = find_skills(text)
    role = infer_role(skills)
    skills_str = ", ".join(skills[:15]) if skills else ""
    return skills_str, role
