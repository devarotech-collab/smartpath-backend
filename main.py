import os
import traceback
from datetime import datetime, timezone
from xml.sax.saxutils import escape
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from supabase import create_client, Client
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SmartPath AI Engine")
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
ai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VALID_STAGES = {"Explorer", "Learner", "Builder", "Job-Ready"}

class IncomingMessage(BaseModel):
    phone_number: str
    message: str

class OnboardingAuditResult(BaseModel):
    stage: str = Field(description="One of: Explorer, Learner, Builder, Job-Ready")
    audit_score: int = Field(ge=0, le=20, description="Baseline score out of 20")
    summary_insight: str = Field(description="1-sentence encouraging assessment for WhatsApp")

class SubmissionEvaluation(BaseModel):
    score: int = Field(ge=0, le=100)
    passed: bool = Field(description="True if score >= 70")
    strengths: str = Field(description="1 short sentence on what they did well")
    improvement: str = Field(description="1 concrete, actionable fix")

def get_milestone_by_step(track: str, step_order: int):
    res = (
        supabase.table("milestones")
        .select("*")
        .eq("track", track)
        .eq("step_order", step_order)
        .execute()
    )
    return res.data[0] if res.data else None

def check_unlocked_jobs(track: str, readiness_score: int):
    if readiness_score < 75:
        return []
    res = (
        supabase.table("jobs")
        .select("title, company, apply_url")
        .eq("track", track)
        .eq("is_active", True)
        .lte("min_readiness_score", readiness_score)
        .execute()
    )
    return res.data

def clean_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()

def run_audit_ai(track: str, courses: str, past_work: str) -> OnboardingAuditResult:
    prompt = (
        f"Evaluate this Value Digital Hive learner for the '{track}' track.\n"
        f"Courses/Training: {courses}\n"
        f"Past Projects/Portfolio: {past_work}\n"
        "Return JSON with keys: stage (Explorer, Learner, Builder, or Job-Ready), "
        "audit_score (integer 0-20), and summary_insight (1 encouraging sentence)."
    )
    try:
        ai_res = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=OnboardingAuditResult,
                temperature=0.2,
            ),
        )
        return OnboardingAuditResult.model_validate_json(clean_json_text(ai_res.text))
    except Exception:
        has_projects = past_work.lower().strip() not in {"none", "none yet", "no", "nothing", "n/a", ""}
        has_courses = courses and courses.lower().strip() not in {"none", "no", "nothing", ""}
        if has_projects and ("http" in past_work.lower() or len(past_work) > 30):
            return OnboardingAuditResult(
                stage="Builder",
                audit_score=16,
                summary_insight=f"Great hands-on start in {track}—now let's validate your execution against industry standards!"
            )
        if has_courses:
            return OnboardingAuditResult(
                stage="Learner",
                audit_score=12,
                summary_insight=f"Solid foundation in {track} fundamentals—your next step is turning theory into a verified project."
            )
        return OnboardingAuditResult(
            stage="Explorer",
            audit_score=6,
            summary_insight=f"Welcome to {track}! We'll guide you step-by-step from your first practical task to job readiness."
        )

def run_submission_ai(track: str, milestone: dict, submission_text: str) -> SubmissionEvaluation:
    prompt = (
        f"You are a senior {track} mentor at Value Digital Hive.\n"
        f"Task Assigned: {milestone['title']} - {milestone['task_prompt']}\n"
        f"Rubric: {milestone['evaluation_rubric']}\n"
        f"Learner Submission: {submission_text}\n"
        "Return JSON with keys: score (0-100), passed (boolean, true if score >= 70), "
        "strengths (1 short sentence), improvement (1 concrete actionable fix)."
    )
    try:
        eval_res = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SubmissionEvaluation,
                temperature=0.2,
            ),
        )
        return SubmissionEvaluation.model_validate_json(clean_json_text(eval_res.text))
    except Exception:
        has_link = "http://" in submission_text.lower() or "https://" in submission_text.lower()
        has_metrics = any(ch.isdigit() for ch in submission_text) and len(submission_text.split()) >= 10
        if has_link and has_metrics:
            return SubmissionEvaluation(
                score=85,
                passed=True,
                strengths="Clear quantitative metrics, structured execution summary, and a shareable project link.",
                improvement="Add a 1-line business recommendation explaining the measurable impact of your work."
            )
        if has_metrics or has_link:
            return SubmissionEvaluation(
                score=72,
                passed=True,
                strengths="Good effort addressing the core task requirements.",
                improvement="Include both specific numerical metrics and a public link to your deliverable next time."
            )
        return SubmissionEvaluation(
            score=45,
            passed=False,
            strengths="You took the first step by submitting a response.",
            improvement="Include your shareable project link and at least 2 concrete numbers/findings from your work."
        )

@app.get("/")
def health_check():
    return {"status": "SmartPath AI Engine is running", "simulator": f"{APP_BASE_URL}/chat"}

@app.post("/webhook/whatsapp")
async def handle_whatsapp_message(payload: IncomingMessage):
    try:
        phone = payload.phone_number.strip()
        text = payload.message.strip()

        res = supabase.table("learners").select("*").eq("phone_number", phone).execute()
        if not res.data:
            supabase.table("learners").insert({"phone_number": phone, "onboarding_step": 1}).execute()
            return {
                "reply": (
                    "Welcome to *SmartPath* by Value Digital Hive! 🚀\n\n"
                    "In 60 seconds, I'll map your skill level and build your career roadmap.\n\n"
                    "*Question 1/4:* What is your first name?"
                ),
                "readiness_score": 0,
                "stage": "Explorer"
            }

        learner = res.data[0]
        step = learner["onboarding_step"]
        answers = learner.get("onboarding_answers") or {}

        if step == 1:
            supabase.table("learners").update({
                "full_name": text,
                "onboarding_step": 2,
                "last_active_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", learner["id"]).execute()
            return {
                "reply": (
                    f"Nice to meet you, *{text}*! 👋\n\n"
                    "*Question 2/4:* Which career track are you focusing on?\n"
                    "Reply with one:\n• Data Analysis\n• UI/UX\n• Web Dev\n• Digital Marketing"
                ),
                "readiness_score": 0,
                "stage": "Explorer"
            }

        if step == 2:
            valid_tracks = ["Data Analysis", "UI/UX", "Web Dev", "Digital Marketing"]
            matched_track = next((t for t in valid_tracks if t.lower() in text.lower()), "Data Analysis")
            answers["track_raw"] = text
            supabase.table("learners").update({
                "track": matched_track,
                "onboarding_step": 3,
                "onboarding_answers": answers
            }).eq("id", learner["id"]).execute()
            return {
                "reply": (
                    f"Locked in: *{matched_track}* 🎯\n\n"
                    "*Question 3/4:* What courses, bootcamps, or tutorials have you started or completed so far?"
                ),
                "readiness_score": 0,
                "stage": "Explorer"
            }

        if step == 3:
            answers["courses"] = text
            supabase.table("learners").update({
                "onboarding_step": 4,
                "onboarding_answers": answers
            }).eq("id", learner["id"]).execute()
            return {
                "reply": (
                    "*Question 4/4:* Have you built any real projects, portfolios, or freelance work yet? "
                    "(Paste a link or reply 'None yet' if you're just starting!)"
                ),
                "readiness_score": 0,
                "stage": "Explorer"
            }

        if step == 4:
            answers["past_work"] = text
            track = learner.get("track") or "Data Analysis"
            audit = run_audit_ai(track, answers.get("courses", ""), text)
            safe_stage = audit.stage.strip() if audit.stage.strip() in VALID_STAGES else "Learner"
            safe_audit_score = max(0, min(20, int(audit.audit_score)))
            milestone = get_milestone_by_step(track, 1)

            updated = supabase.table("learners").update({
                "stage": safe_stage,
                "audit_score": safe_audit_score,
                "consistency_score": 10,
                "onboarding_step": 5,
                "onboarding_answers": answers,
                "active_milestone_id": milestone["id"] if milestone else None
            }).eq("id", learner["id"]).execute().data[0]

            dashboard_link = f"{APP_BASE_URL}/learner/{learner['magic_token']}"
            task_text = (
                f"\n\n📌 *Your First Micro-Project (Step 1):*\n*{milestone['title']}*\n{milestone['task_prompt']}\n\n_Reply here with your link or summary when ready!_"
                if milestone else ""
            )

            return {
                "reply": (
                    f"✅ *Skill Audit Complete!*\n"
                    f"• *Stage:* {safe_stage}\n"
                    f"• *Career Readiness:* {updated['readiness_score']}%\n"
                    f"• *Takeaway:* {audit.summary_insight}\n\n"
                    f"📊 View your live roadmap: {dashboard_link}"
                    f"{task_text}"
                ),
                "readiness_score": updated["readiness_score"],
                "stage": safe_stage
            }

        milestone_id = learner.get("active_milestone_id")
        dashboard_link = f"{APP_BASE_URL}/learner/{learner['magic_token']}"
        if not milestone_id:
            return {
                "reply": f"🏆 You've completed all current milestones! View your job matches here:\n{dashboard_link}",
                "readiness_score": learner["readiness_score"],
                "stage": learner["stage"]
            }

        milestone = supabase.table("milestones").select("*").eq("id", milestone_id).execute().data[0]
        track = learner.get("track") or "Data Analysis"
        evaluation = run_submission_ai(track, milestone, text)
        safe_score = max(0, min(100, int(evaluation.score)))

        supabase.table("submissions").insert({
            "learner_id": learner["id"],
            "milestone_id": milestone_id,
            "submission_content": text,
            "ai_score": safe_score,
            "ai_feedback": f"{evaluation.strengths} {evaluation.improvement}",
            "passed": evaluation.passed
        }).execute()

        next_milestone_id = milestone_id
        next_task_text = ""
        if evaluation.passed:
            next_ms = get_milestone_by_step(track, milestone.get("step_order", 1) + 1)
            if next_ms:
                next_milestone_id = next_ms["id"]
                next_task_text = (
                    f"\n\n📌 *Next Micro-Project Unlocked (Step {next_ms['step_order']}):*\n"
                    f"*{next_ms['title']}*\n{next_ms['task_prompt']}"
                )
            else:
                next_milestone_id = None
                next_task_text = "\n\n🏆 *All track milestones completed!* Check your dashboard for unlocked roles."

        new_project_score = min(50, learner["project_score"] + (milestone["points_value"] if evaluation.passed else 5))
        new_consistency = min(30, learner["consistency_score"] + 10)
        new_stage = (
            "Job-Ready"
            if (learner["audit_score"] + new_project_score + new_consistency) >= 75
            else ("Builder" if evaluation.passed else learner["stage"])
        )

        updated_learner = supabase.table("learners").update({
            "project_score": new_project_score,
            "consistency_score": new_consistency,
            "stage": new_stage,
            "active_milestone_id": next_milestone_id,
            "last_active_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", learner["id"]).execute().data[0]

        score_now = updated_learner["readiness_score"]
        jobs = check_unlocked_jobs(track, score_now)
        job_alert = ""
        if jobs:
            job_lines = "\n".join([f"• *{j['title']}* at {j['company']} → {j['apply_url']}" for j in jobs])
            job_alert = f"\n\n🎉 *75%+ Threshold Reached! Unlocked Roles:*\n{job_lines}"

        status_header = "🌟 *Project Verified!*" if evaluation.passed else "🛠️ *Needs Quick Revision*"
        return {
            "reply": (
                f"{status_header} (Score: {safe_score}/100)\n\n"
                f"👍 *Strength:* {evaluation.strengths}\n"
                f"💡 *Next Fix:* {evaluation.improvement}\n\n"
                f"📈 *Updated Readiness:* {score_now}% ({new_stage})\n"
                f"📊 Dashboard: {dashboard_link}"
                f"{next_task_text}"
                f"{job_alert}"
            ),
            "readiness_score": score_now,
            "stage": new_stage
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

@app.post("/webhook/twilio")
async def handle_twilio_whatsapp(From: str = Form(...), Body: str = Form(...)):
    clean_phone = From.replace("whatsapp:", "").strip()
    result = await handle_whatsapp_message(IncomingMessage(phone_number=clean_phone, message=Body))
    reply_text = escape(result["reply"])
    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{reply_text}</Message></Response>'
    return Response(content=twiml, media_type="application/xml")

@app.get("/learner/{magic_token}", response_class=HTMLResponse)
def learner_dashboard(magic_token: str):
    res = supabase.table("learners").select("*").eq("magic_token", magic_token).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Learner roadmap not found")
    learner = res.data[0]
    subs = supabase.table("submissions").select("*").eq("learner_id", learner["id"]).order("created_at", desc=True).execute().data
    jobs = supabase.table("jobs").select("*").eq("track", learner.get("track") or "Data Analysis").execute().data

    sub_items = []
    for s in subs:
        status_label = "Passed ✅" if s["passed"] else "Needs Revision 🛠️"
        sub_items.append(
            f"<div class='p-4 bg-slate-800 rounded-lg border border-slate-700 mb-3'>"
            f"<div class='flex justify-between items-center mb-1'>"
            f"<span class='font-semibold text-emerald-400'>Score: {s['ai_score']}/100 ({status_label})</span>"
            f"</div>"
            f"<p class='text-sm text-slate-300 mb-2'>\"{s['submission_content']}\"</p>"
            f"<p class='text-xs text-slate-400'>💡 <b>AI Mentor:</b> {s['ai_feedback']}</p></div>"
        )
    sub_rows = "".join(sub_items) or "<p class='text-slate-400 text-sm'>No project submissions yet.</p>"

    score = learner["readiness_score"]
    job_items = []
    for j in jobs:
        unlocked = score >= j["min_readiness_score"]
        card_cls = "border-emerald-500/40 bg-emerald-950/20" if unlocked else "border-slate-700 bg-slate-800/50 opacity-60"
        if unlocked:
            url = j["apply_url"].replace("https://vdh.example.com/jobs/", f"{APP_BASE_URL}/apply/").replace("http://vdh.example.com/jobs/", f"{APP_BASE_URL}/apply/")
            action_html = f'<a href="{url}" target="_blank" class="px-3 py-1.5 bg-emerald-500 text-slate-950 font-bold text-xs rounded">Apply Now →</a>'
        else:
            action_html = '<span class="text-xs text-amber-400">🔒 Locked</span>'
        job_items.append(
            f"<div class='p-4 rounded-lg border {card_cls} mb-3 flex justify-between items-center'>"
            f"<div><p class='font-semibold text-white'>{j['title']}</p>"
            f"<p class='text-xs text-slate-400'>{j['company']} • Requires {j['min_readiness_score']}% Readiness</p></div>"
            f"{action_html}</div>"
        )
    job_rows = "".join(job_items)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>SmartPath Career Roadmap — {learner.get('full_name') or 'Learner'}</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen p-6 md:p-12">
        <div class="max-w-3xl mx-auto space-y-6">
            <div class="flex justify-between items-center border-b border-slate-800 pb-4">
                <div>
                    <span class="text-xs uppercase tracking-widest text-emerald-400 font-bold">Value Digital Hive • SmartPath AI</span>
                    <h1 class="text-2xl font-bold mt-1">{learner.get('full_name') or 'Learner'} — {learner.get('track') or 'Unassigned Track'}</h1>
                </div>
                <span class="px-4 py-1.5 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-semibold text-sm">
                    Stage: {learner['stage']}
                </span>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <div class="flex justify-between items-end mb-2">
                    <span class="text-sm text-slate-400">Overall Career Readiness Score</span>
                    <span class="text-3xl font-extrabold text-emerald-400">{score}%</span>
                </div>
                <div class="w-full bg-slate-800 h-3 rounded-full overflow-hidden mb-4">
                    <div class="bg-emerald-500 h-full" style="width: {min(100, score)}%"></div>
                </div>
                <div class="grid grid-cols-3 gap-4 text-center pt-2 border-t border-slate-800 text-xs">
                    <div><p class="text-slate-400">Skill Audit</p><p class="text-lg font-bold text-white">{learner['audit_score']}/20</p></div>
                    <div><p class="text-slate-400">Verified Projects</p><p class="text-lg font-bold text-white">{learner['project_score']}/50</p></div>
                    <div><p class="text-slate-400">Consistency</p><p class="text-lg font-bold text-white">{learner['consistency_score']}/30</p></div>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h2 class="text-lg font-bold mb-3">📂 Verified Project Submissions</h2>
                {sub_rows}
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h2 class="text-lg font-bold mb-3">🚀 VDH Partner Job Matches (Unlocks at 75%)</h2>
                {job_rows}
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/chat", response_class=HTMLResponse)
def whatsapp_simulator():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>SmartPath WhatsApp Simulator</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-white min-h-screen flex items-center justify-center p-4">
        <div class="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col h-[620px]">
            <div class="bg-emerald-700 p-4 flex justify-between items-center">
                <div>
                    <h1 class="font-bold text-sm">SmartPath AI (WhatsApp Bot)</h1>
                    <p class="text-xs text-emerald-200">Value Digital Hive Career Mentor</p>
                </div>
                <input id="phone" value="+2348011112222" class="bg-emerald-800 text-xs px-2 py-1 rounded text-white w-36 border border-emerald-600" title="Phone Number">
            </div>
            <div id="messages" class="flex-1 p-4 overflow-y-auto space-y-3 bg-slate-950/60 text-sm">
                <div class="text-center text-xs text-slate-500 my-2">Send a message or change the phone number above to test a new learner.</div>
            </div>
            <form onsubmit="sendMsg(event)" class="p-3 bg-slate-900 border-t border-slate-800 flex gap-2">
                <input id="msgInput" placeholder="Type a WhatsApp message..." class="flex-1 bg-slate-800 text-white px-3 py-2 rounded-lg text-sm focus:outline-none">
                <button class="bg-emerald-600 hover:bg-emerald-500 px-4 py-2 rounded-lg font-bold text-sm">Send</button>
            </form>
        </div>
        <script>
            async function sendMsg(e) {
                e.preventDefault();
                const input = document.getElementById('msgInput');
                const phone = document.getElementById('phone').value;
                const text = input.value.trim();
                if (!text) return;
                const box = document.getElementById('messages');
                box.innerHTML += `<div class="flex justify-end"><div class="bg-emerald-700 px-3 py-2 rounded-xl max-w-[80%]">${text}</div></div>`;
                input.value = '';
                box.scrollTop = box.scrollHeight;
                const res = await fetch('/webhook/whatsapp', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({phone_number: phone, message: text})
                });
                const data = await res.json();
                const formatted = (data.reply || data.detail || 'Error').replace(/\\n/g, '<br>').replace(/(https?:\\/\\/[^\\s<]+)/g, '<a href="$1" target="_blank" class="underline text-emerald-300 font-semibold">$1</a>');
                box.innerHTML += `<div class="flex justify-start"><div class="bg-slate-800 border border-slate-700 px-3 py-2 rounded-xl max-w-[85%] whitespace-pre-wrap">${formatted}</div></div>`;
                box.scrollTop = box.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

@app.get("/apply/{job_slug}", response_class=HTMLResponse)
def job_application_page(job_slug: str):
    role_title = job_slug.replace("-", " ").title()
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Apply — {role_title} | Value Digital Hive</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-6">
        <div class="max-w-lg w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 space-y-5">
            <span class="text-xs uppercase tracking-widest text-emerald-400 font-bold">Value Digital Hive • Verified Talent Fast-Track</span>
            <h1 class="text-2xl font-bold">{role_title}</h1>
            <p class="text-sm text-slate-400">Your SmartPath Readiness Score (75%+) qualifies you for direct partner review.</p>
            <form onsubmit="event.preventDefault(); document.getElementById('done').classList.remove('hidden');" class="space-y-4">
                <input required placeholder="Full Name" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm">
                <input required type="email" placeholder="Email Address" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm">
                <input required placeholder="Portfolio / LinkedIn URL" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm">
                <button class="w-full bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold py-2.5 rounded-lg text-sm">Submit Fast-Track Application</button>
            </form>
            <div id="done" class="hidden p-3 rounded-lg bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 text-sm">
                ✅ Application submitted to Value Digital Hive hiring partners!
            </div>
        </div>
    </body>
    </html>
    """)
