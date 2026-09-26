import asyncio, base64, hashlib, hmac, json, os, shutil, sqlite3, subprocess, uuid
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
BASE=Path(__file__).parent
DB_PATH=BASE/os.getenv("DATABASE_PATH","data/storymotion.db")
OUT=BASE/"outputs"; OUT.mkdir(exist_ok=True)
DB_PATH.parent.mkdir(exist_ok=True)

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
STORY_MODEL=os.getenv("STORY_MODEL","gemini-2.5-flash")
VIDEO_MODEL=os.getenv("VIDEO_MODEL","veo-3.1-generate-preview")
FREE_CREDITS=int(os.getenv("FREE_CREDITS","60"))
MIDTRANS_SERVER_KEY=os.getenv("MIDTRANS_SERVER_KEY","")
MIDTRANS_CLIENT_KEY=os.getenv("MIDTRANS_CLIENT_KEY","")
MIDTRANS_PROD=os.getenv("MIDTRANS_IS_PRODUCTION","false").lower()=="true"
BASE_URL=os.getenv("APP_BASE_URL","http://127.0.0.1:8000")

app=FastAPI(title="StoryMotion AI",version="3.0.0")
app.mount("/static",StaticFiles(directory=BASE/"static"),name="static")
app.mount("/outputs",StaticFiles(directory=OUT),name="outputs")

PACKAGES={"starter":(120,19000),"creator":(400,49000),"studio":(1000,99000)}

def db():
    c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
      id TEXT PRIMARY KEY, email TEXT, name TEXT, credits INTEGER NOT NULL DEFAULT 60,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS projects(
      id TEXT PRIMARY KEY, user_id TEXT, title TEXT, idea TEXT, genre TEXT, style TEXT,
      duration INTEGER, aspect_ratio TEXT, plan_json TEXT, status TEXT DEFAULT 'draft',
      video_url TEXT, subtitle_url TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS payments(
      order_id TEXT PRIMARY KEY, user_id TEXT, package_id TEXT, credits INTEGER,
      amount INTEGER, status TEXT DEFAULT 'pending', created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """); c.commit(); return c

def ensure_user(user_id="demo-user",email="demo@storymotion.local",name="Demo Creator"):
    c=db(); row=c.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone()
    if not row: c.execute("INSERT INTO users(id,email,name,credits) VALUES(?,?,?,?)",(user_id,email,name,FREE_CREDITS)); c.commit()
    return c.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone()

class StoryRequest(BaseModel):
    idea:str=Field(min_length=5,max_length=5000)
    genre:str="Drama"; style:str="3D cinematic animation"
    duration:int=Field(default=60,ge=30,le=180)
    aspect_ratio:str="9:16"; language:str="Indonesian"
    characters:list[dict]=[]
class GenerateRequest(StoryRequest): project_id:Optional[str]=None
class RegenerateRequest(BaseModel): feedback:str=""
class CheckoutRequest(BaseModel): package_id:str

def user_id_from_header(x_user_id:Optional[str]):
    # Prototype: replace with verified Firebase ID token in production.
    return x_user_id or "demo-user"

def scene_count(d): return max(4,min(22,round(d/8)))

def extract_json(t):
    t=t.strip()
    if "```" in t:
        t=t.split("```",2)[1]
        if t.lstrip().startswith("json"): t=t.lstrip()[4:]
    a,b=t.find("{"),t.rfind("}")
    if a<0 or b<0: raise ValueError("AI did not return JSON")
    return json.loads(t[a:b+1])

async def create_plan(req):
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY belum diatur.")
    from google import genai
    client=genai.Client(api_key=GEMINI_API_KEY)
    n=scene_count(req.duration)
    chars="\n".join(f"- {x.get('name','')}: {x.get('description','')}; wardrobe: {x.get('wardrobe','')}" for x in req.characters) or "Create original characters."
    prompt=f"""You are StoryMotion AI's showrunner, screenwriter and 3D animation director.
Create a production-ready {n}-scene short film. REAL MOVING ANIMATION, never slideshow.
IDEA: {req.idea}
GENRE: {req.genre}; STYLE: {req.style}; DURATION: {req.duration}s; ASPECT: {req.aspect_ratio}; LANGUAGE: {req.language}
USER CHARACTERS:
{chars}
Keep character face, hair, proportions, clothing and props consistent. Each scene ~8 seconds,
with continuous physical motion, camera movement, environment motion, lighting and emotion.
Use original characters/settings. No celebrities or copyrighted characters.
Return ONLY JSON:
{{"title":"string","logline":"string","character_bible":[{{"id":"string","name":"string","description":"string","wardrobe":"string","personality":"string"}}],
"scenes":[{{"number":1,"duration_seconds":8,"setting":"string","action":"string","camera":"string","lighting":"string",
"dialogue":"string","narration":"string","video_prompt":"string","subtitle_lines":["string"]}}]}}"""
    r=await asyncio.to_thread(client.models.generate_content,model=STORY_MODEL,contents=prompt)
    return extract_json(r.text)

def video_prompt(plan,scene,ratio,feedback=""):
    cb="\n".join(f"- {c['name']}: {c['description']}; wardrobe: {c['wardrobe']}; personality: {c['personality']}" for c in plan["character_bible"])
    return f"""Create one continuous high-quality 3D animated cinematic shot.
FORMAT {ratio}. Feature-film 3D animation, expressive faces, natural body mechanics,
cloth/hair motion, detailed environment, cinematic depth of field, volumetric lighting.
CHARACTER BIBLE:
{cb}
SCENE: {scene['video_prompt']}
ACTION: {scene['action']}
CAMERA: {scene['camera']}
LIGHTING: {scene['lighting']}
DIALOGUE/AUDIO: {scene.get('dialogue','')} {scene.get('narration','')}
Keep faces, hair, body proportions, wardrobe and props consistent.
Continuous visible movement throughout. NOT a still, storyboard, collage or slideshow.
No watermark, random text, burned subtitles, extra characters, deformed hands or face changes.
REGENERATION FEEDBACK: {feedback or 'None'}"""

async def make_clip(plan,scene,ratio,path,feedback=""):
    from google import genai
    from google.genai import types
    client=genai.Client(api_key=GEMINI_API_KEY)
    op=await asyncio.to_thread(client.models.generate_videos,model=VIDEO_MODEL,
        prompt=video_prompt(plan,scene,ratio,feedback),
        config=types.GenerateVideosConfig(aspect_ratio=ratio))
    while not op.done:
        await asyncio.sleep(8); op=await asyncio.to_thread(client.operations.get,op)
    if getattr(op,"error",None): raise RuntimeError(str(op.error))
    v=op.response.generated_videos[0].video
    await asyncio.to_thread(client.files.download,file=v,download_path=str(path))

def concat(paths,out):
    if not shutil.which("ffmpeg"): raise RuntimeError("FFmpeg tidak ditemukan.")
    f=out.parent/"concat.txt"; f.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in paths))
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(f),"-c","copy",str(out)],check=True,capture_output=True)
    f.unlink(missing_ok=True)

def srt(plan,path):
    def ts(x): return f"00:{x//60:02d}:{x%60:02d},000"
    cur=0; out=[]
    for sc in plan["scenes"]:
        txt=sc.get("dialogue") or sc.get("narration") or ""
        if txt.strip(): out.append(f"{len(out)+1}\n{ts(cur)} --> {ts(cur+int(sc.get('duration_seconds',8)))}\n{txt.strip()}\n")
        cur+=int(sc.get("duration_seconds",8))
    path.write_text("\n".join(out),encoding="utf-8")

jobs={}
async def run(project_id,req,user_id):
    c=db(); row=c.execute("SELECT * FROM projects WHERE id=?",(project_id,)).fetchone()
    try:
        c.execute("UPDATE projects SET status='planning' WHERE id=?",(project_id,)); c.commit()
        plan=await create_plan(req)
        c.execute("UPDATE projects SET title=?,plan_json=?,status='generating' WHERE id=?",(plan["title"],json.dumps(plan),project_id)); c.commit()
        d=OUT/project_id; d.mkdir(exist_ok=True); clips=[]
        for i,sc in enumerate(plan["scenes"],1):
            jobs[project_id]={"status":"generating","scene":i,"total":len(plan["scenes"])}
            p=d/f"scene_{i:02d}.mp4"; await make_clip(plan,sc,req.aspect_ratio,p); clips.append(p)
        final=d/"storymotion_movie.mp4"; concat(clips,final)
        sp=d/"subtitles.srt"; srt(plan,sp)
        c.execute("UPDATE projects SET status='completed',video_url=?,subtitle_url=? WHERE id=?",
                  (f"/outputs/{project_id}/storymotion_movie.mp4",f"/outputs/{project_id}/subtitles.srt",project_id)); c.commit()
        jobs[project_id]={"status":"completed","scene":len(clips),"total":len(clips)}
    except Exception as e:
        c.execute("UPDATE projects SET status='failed' WHERE id=?",(project_id,)); c.commit()
        jobs[project_id]={"status":"failed","error":str(e)}

@app.get("/")
async def home(): return FileResponse(BASE/"static/index.html")
@app.get("/api/health")
async def health(): return {"ok":True,"version":"3.0.0","video_model":VIDEO_MODEL}

@app.get("/api/me")
async def me(x_user_id:Optional[str]=Header(None)):
    u=ensure_user(user_id_from_header(x_user_id)); return dict(u)

@app.get("/api/projects")
async def projects(x_user_id:Optional[str]=Header(None)):
    uid=user_id_from_header(x_user_id); ensure_user(uid)
    c=db(); return [dict(r) for r in c.execute("SELECT id,title,status,duration,aspect_ratio,video_url,created_at FROM projects WHERE user_id=? ORDER BY created_at DESC",(uid,)).fetchall()]

@app.get("/api/projects/{pid}")
async def project(pid,x_user_id:Optional[str]=Header(None)):
    uid=user_id_from_header(x_user_id); c=db(); r=c.execute("SELECT * FROM projects WHERE id=? AND user_id=?",(pid,uid)).fetchone()
    if not r: raise HTTPException(404,"Project tidak ditemukan")
    d=dict(r); d["plan"]=json.loads(d.pop("plan_json") or "{}"); return d

@app.post("/api/plan")
async def plan(req:StoryRequest):
    try:return await create_plan(req)
    except Exception as e: raise HTTPException(500,str(e))

@app.post("/api/generate")
async def generate(req:GenerateRequest,x_user_id:Optional[str]=Header(None)):
    uid=user_id_from_header(x_user_id); u=ensure_user(uid); cost=scene_count(req.duration)
    if u["credits"]<cost: raise HTTPException(402,f"Kredit kurang. Butuh sekitar {cost}.")
    pid=req.project_id or uuid.uuid4().hex[:12]
    c=db()
    c.execute("UPDATE users SET credits=credits-? WHERE id=?",(cost,uid))
    c.execute("""INSERT OR REPLACE INTO projects(id,user_id,title,idea,genre,style,duration,aspect_ratio,status)
                 VALUES(?,?,?,?,?,?,?,?,?)""",(pid,uid,"Untitled",req.idea,req.genre,req.style,req.duration,req.aspect_ratio,"queued"))
    c.commit()
    asyncio.create_task(run(pid,req,uid))
    return {"project_id":pid,"credits_left":u["credits"]-cost}

@app.get("/api/projects/{pid}/status")
async def status(pid): return jobs.get(pid,{"status":"queued"})

@app.post("/api/projects/{pid}/scenes/{scene}/regenerate")
async def regen(pid,scene:int,req:RegenerateRequest,x_user_id:Optional[str]=Header(None)):
    uid=user_id_from_header(x_user_id); c=db(); r=c.execute("SELECT * FROM projects WHERE id=? AND user_id=?",(pid,uid)).fetchone()
    if not r: raise HTTPException(404,"Project tidak ditemukan")
    u=ensure_user(uid)
    if u["credits"]<1: raise HTTPException(402,"Kredit tidak cukup")
    plan=json.loads(r["plan_json"]); 
    if scene<1 or scene>len(plan["scenes"]): raise HTTPException(400,"Scene tidak valid")
    c.execute("UPDATE users SET credits=credits-1 WHERE id=?",(uid,)); c.commit()
    p=OUT/pid/f"scene_{scene:02d}.mp4"; await make_clip(plan,plan["scenes"][scene-1],r["aspect_ratio"],p,req.feedback)
    paths=[OUT/pid/f"scene_{i:02d}.mp4" for i in range(1,len(plan["scenes"])+1)]
    concat(paths,OUT/pid/"storymotion_movie.mp4")
    return {"ok":True,"video_url":f"/outputs/{pid}/storymotion_movie.mp4","credits_left":u["credits"]-1}

@app.post("/api/payments/create")
async def payment(req:CheckoutRequest,x_user_id:Optional[str]=Header(None)):
    uid=user_id_from_header(x_user_id)
    if req.package_id not in PACKAGES: raise HTTPException(400,"Package tidak valid")
    if not MIDTRANS_SERVER_KEY: raise HTTPException(503,"Midtrans belum dikonfigurasi")
    cr,amount=PACKAGES[req.package_id]; order=f"SM-{uuid.uuid4().hex[:16]}"
    c=db(); c.execute("INSERT INTO payments(order_id,user_id,package_id,credits,amount) VALUES(?,?,?,?,?)",(order,uid,req.package_id,cr,amount)); c.commit()
    host="app.midtrans.com" if MIDTRANS_PROD else "app.sandbox.midtrans.com"
    auth=base64.b64encode((MIDTRANS_SERVER_KEY+":").encode()).decode()
    res=requests.post(f"https://{host}/snap/v1/transactions",headers={"Authorization":f"Basic {auth}","Content-Type":"application/json"},
      json={"transaction_details":{"order_id":order,"gross_amount":amount},"item_details":[{"id":req.package_id,"price":amount,"quantity":1,"name":f"StoryMotion {req.package_id} credits"}]},timeout=30)
    if not res.ok: raise HTTPException(502,res.text)
    data=res.json(); return {"order_id":order,"token":data["token"],"redirect_url":data.get("redirect_url"),"client_key":MIDTRANS_CLIENT_KEY}

@app.post("/api/payments/notification")
async def notification(payload:dict):
    order=payload.get("order_id"); status=payload.get("transaction_status")
    c=db(); p=c.execute("SELECT * FROM payments WHERE order_id=?",(order,)).fetchone()
    if not p: return {"ok":True}
    # Production MUST verify signature/status with Midtrans before crediting.
    if status in ("settlement","capture") and p["status"]!="paid":
        c.execute("UPDATE payments SET status='paid' WHERE order_id=?",(order,))
        c.execute("UPDATE users SET credits=credits+? WHERE id=?",(p["credits"],p["user_id"])); c.commit()
    elif status in ("cancel","deny","expire"): c.execute("UPDATE payments SET status=? WHERE order_id=?",(status,order)); c.commit()
    return {"ok":True}

@app.get("/api/config/payment")
async def payconfig(): return {"enabled":bool(MIDTRANS_SERVER_KEY and MIDTRANS_CLIENT_KEY),"production":MIDTRANS_PROD,"client_key":MIDTRANS_CLIENT_KEY}

@app.get("/api/packages")
async def packages(): return [{"id":k,"credits":v[0],"price_idr":v[1]} for k,v in PACKAGES.items()]
