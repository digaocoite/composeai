import os, json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

# Load env (.env must contain OPENAI_API_KEY=sk-...)
load_dotenv(".env")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

app = FastAPI(title="Mizzou Spanish Composition Checker")

# Serve /static and /
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/ping")
def ping():
    return {"ok": True, "app": "mizzou-span-1200"}

@app.post("/api/check")
async def check(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    prompt = f"""
You are a Spanish instructor at the University of Missouri (SPAN 1200).
The student wrote about their last birthday. Correct grammar, spelling, and accent marks in Spanish,
with emphasis on past tense (preterite) regular and irregular verbs (aim for ~6 of each).
Then provide explanations in English for the main corrections (bullet points, concise).
Return JSON with keys: "corrected_text", "explanations_md".
Student text:
{text}
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
