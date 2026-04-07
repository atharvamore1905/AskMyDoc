import streamlit as st
import tempfile, os, io, re, torch, json, requests
from transformers.models.t5 import T5Tokenizer as AutoTokenizer, T5ForConditionalGeneration as AutoModelForSeq2SeqLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import streamlit.components.v1 as components
from gtts import gTTS
from deep_translator import GoogleTranslator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AskMyDoc", page_icon="📄", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');"
    "html,body,[class*='css']{font-family:'Inter',sans-serif;}"
    ".stApp{background:#0d0d0d;color:#f0ede6;}"
    ".main .block-container{padding-top:1.5rem;}"

    # Sidebar
    "[data-testid='stSidebar']{background:#080808 !important;border-right:1px solid #161616 !important;}"
    "[data-testid='stSidebar'] section{padding:1.2rem !important;}"

    # File uploader
    "[data-testid='stFileUploader']{background:#111;border:1.5px dashed #2a2a2a;border-radius:12px;padding:0.6rem;}"
    "[data-testid='stFileUploaderDropzoneInstructions']{color:#555 !important;}"

    # Text input
    "[data-testid='stTextInput'] input{background:#111 !important;border:1.5px solid #222 !important;"
    "border-radius:10px !important;color:#f0ede6 !important;font-size:0.95rem !important;padding:0.75rem 1rem !important;}"
    "[data-testid='stTextInput'] input:focus{border-color:#c8f04e !important;box-shadow:0 0 0 2px rgba(200,240,78,0.08) !important;}"
    "[data-testid='stTextInput'] input::placeholder{color:#444 !important;}"

    # FIX 1 — Selectbox fully styled so it's visible on dark background
    "[data-testid='stSelectbox']{margin-bottom:0.2rem;}"
    "[data-testid='stSelectbox'] > div > div{background:#111 !important;"
    "border:1.5px solid #2a2a2a !important;border-radius:10px !important;"
    "color:#f0ede6 !important;font-size:0.88rem !important;}"
    "[data-testid='stSelectbox'] svg{fill:#c8f04e !important;}"
    "[data-baseweb='select'] span{color:#f0ede6 !important;}"
    "[data-baseweb='popover'] ul{background:#111 !important;border:1px solid #2a2a2a !important;border-radius:10px !important;}"
    "[data-baseweb='popover'] li{color:#f0ede6 !important;background:#111 !important;}"
    "[data-baseweb='popover'] li:hover{background:#1e1e1e !important;color:#c8f04e !important;}"

    # Buttons
    ".stButton>button{background:#c8f04e !important;color:#0d0d0d !important;font-family:'Syne',sans-serif !important;"
    "font-weight:700 !important;border:none !important;border-radius:10px !important;"
    "padding:0.6rem 1.4rem !important;font-size:0.85rem !important;letter-spacing:0.03em !important;"
    "transition:opacity 0.15s,transform 0.1s !important;}"
    ".stButton>button:hover{opacity:0.82 !important;transform:translateY(-1px) !important;}"
    ".stButton>button:active{transform:translateY(0) !important;}"

    # Tabs
    ".stTabs [data-baseweb='tab-list']{background:#0f0f0f;border-radius:12px;padding:5px;gap:3px;"
    "border:1px solid #1a1a1a;width:fit-content;}"
    ".stTabs [data-baseweb='tab']{background:transparent;color:#555;border-radius:9px;font-family:'Syne',sans-serif;"
    "font-size:0.8rem;font-weight:700;letter-spacing:0.05em;padding:0.45rem 1.1rem;transition:all 0.15s;}"
    ".stTabs [aria-selected='true']{background:#1e1e1e !important;color:#c8f04e !important;}"
    ".stTabs [data-baseweb='tab-highlight']{display:none !important;}"
    ".stTabs [data-baseweb='tab-border']{display:none !important;}"
    ".stTabs [data-baseweb='tab-panel']{padding-top:1.5rem !important;}"

    # Expander
    ".streamlit-expanderHeader{background:#111 !important;border-radius:8px !important;color:#888 !important;"
    "font-size:0.82rem !important;border:1px solid #1e1e1e !important;}"
    ".streamlit-expanderContent{background:#0d0d0d !important;border:1px solid #1e1e1e !important;"
    "border-top:none !important;border-radius:0 0 8px 8px !important;}"

    # Alerts
    "[data-testid='stAlert']{border-radius:10px !important;border:none !important;}"

    # Divider
    "hr{border-color:#181818 !important;margin:1.2rem 0 !important;}"

    # Custom classes
    ".sidebar-label{font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;"
    "color:#444;text-transform:uppercase;margin-bottom:0.5rem;margin-top:0.2rem;display:block;}"
    ".stat-row{display:flex;gap:0.5rem;margin:0.6rem 0;}"
    ".stat-pill{flex:1;background:#111;border:1px solid #1e1e1e;border-radius:8px;padding:0.5rem 0.7rem;"
    "font-size:0.78rem;color:#555;text-align:center;}"
    ".stat-pill strong{display:block;font-size:1.1rem;color:#c8f04e;font-family:'Syne',sans-serif;font-weight:700;}"
    ".page-title{font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#f0ede6;"
    "letter-spacing:-0.5px;margin:0 0 0.2rem 0;}"
    ".page-sub{font-size:0.88rem;color:#555;font-weight:300;margin:0;}"
    ".accent{color:#c8f04e;}"
    ".welcome-card{background:#0f0f0f;border:1px solid #191919;border-radius:16px;"
    "padding:3rem 2rem;text-align:center;margin-top:1rem;}"
    # FIX 2 — 3-column grid so Notes fits naturally
    ".feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-top:1.5rem;}"
    ".feature-item{background:#111;border:1px solid #1e1e1e;border-radius:10px;padding:1rem;"
    "font-size:0.83rem;color:#666;text-align:left;}"
    ".feature-item span{display:block;font-size:1.2rem;margin-bottom:0.3rem;}"
    ".feature-item strong{color:#f0ede6;font-family:'Syne',sans-serif;font-size:0.83rem;display:block;}"
    ".info-card{background:#0f0f0f;border:1px solid #191919;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem;}"
    ".info-card-label{font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;"
    "color:#c8f04e;text-transform:uppercase;margin-bottom:0.4rem;}"
    ".info-card-text{font-size:0.85rem;color:#555;margin:0;}"
    ".chat-bubble-user{background:#141f02;border:1px solid #1e2d04;border-radius:14px 14px 4px 14px;"
    "padding:0.9rem 1.2rem;margin:0.6rem 0;margin-left:10%;}"
    ".chat-bubble-bot{background:#111;border:1px solid #1e1e1e;border-radius:14px 14px 14px 4px;"
    "padding:0.9rem 1.2rem;margin:0.6rem 0;margin-right:10%;}"
    ".chat-role{font-family:'Syne',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
    "text-transform:uppercase;margin-bottom:0.4rem;}"
    ".chat-role-user{color:#c8f04e;}"
    ".chat-role-bot{color:#555;}"
    ".chat-text{font-size:0.93rem;color:#f0ede6;line-height:1.65;}"
    ".src-chip{display:inline-block;background:#161616;border:1px solid #252525;border-radius:6px;"
    "padding:0.18rem 0.55rem;font-size:0.7rem;color:#666;margin:0.15rem;cursor:default;}"
    ".src-chip:hover{border-color:#c8f04e;color:#c8f04e;}"
    ".bullet-item{display:flex;align-items:flex-start;gap:0.75rem;padding:0.75rem 1rem;"
    "background:#0f0f0f;border:1px solid #191919;border-radius:10px;margin:0.45rem 0;}"
    ".bullet-dot{color:#c8f04e;font-size:1rem;margin-top:0.05rem;flex-shrink:0;}"
    ".bullet-text{font-size:0.91rem;color:#d0cdc6;line-height:1.55;}"
    ".quiz-card{background:#0f0f0f;border:1px solid #191919;border-radius:12px;"
    "padding:1.1rem 1.3rem;margin:0.7rem 0;}"
    ".quiz-num{font-family:'Syne',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.12em;"
    "color:#555;text-transform:uppercase;margin-bottom:0.4rem;}"
    ".quiz-q{font-size:0.93rem;color:#f0ede6;font-weight:500;line-height:1.5;}"
    ".quiz-ans{background:#0d1600;border-left:3px solid #c8f04e;border-radius:0 8px 8px 0;"
    "padding:0.65rem 1rem;font-size:0.88rem;color:#c8f04e;margin-top:0.5rem;line-height:1.5;}"
    ".mermaid-wrap{background:#080808;border:1px solid #1a1a1a;border-radius:12px;padding:1rem;overflow:hidden;}"
    ".mic-hint{font-size:0.78rem;color:#444;margin:0.4rem 0 0.6rem 0;}"
    # Notes specific
    ".note-qr{display:flex;align-items:flex-start;gap:0.9rem;background:#0f0f0f;"
    "border:1px solid #191919;border-radius:10px;padding:0.8rem 1rem;margin:0.45rem 0;}"
    ".note-qr-num{font-family:'Syne',sans-serif;font-size:0.72rem;font-weight:800;"
    "color:#c8f04e;min-width:1.6rem;text-align:center;padding-top:0.05rem;}"
    ".note-qr-body{flex:1;}"
    ".note-qr-title{font-family:'Syne',sans-serif;font-size:0.78rem;font-weight:700;"
    "color:#c8f04e;letter-spacing:0.04em;margin-bottom:0.25rem;text-transform:uppercase;}"
    ".note-qr-fact{font-size:0.9rem;color:#d0cdc6;line-height:1.55;}"
    ".note-def{background:#0f0f0f;border:1px solid #191919;"
    "border-left:3px solid #c8f04e;border-radius:0 10px 10px 0;padding:0.75rem 1rem;margin:0.4rem 0;}"
    ".note-def-term{font-family:'Syne',sans-serif;font-size:0.82rem;font-weight:700;color:#c8f04e;}"
    ".note-def-text{font-size:0.88rem;color:#d0cdc6;margin-top:0.15rem;line-height:1.5;}"
    ".note-formula{background:#0a0f00;border:1px solid #1a2200;border-radius:10px;"
    "padding:0.7rem 1rem;margin:0.4rem 0;font-family:monospace;font-size:0.9rem;color:#c8f04e;}"
    ".note-formula-label{font-size:0.7rem;color:#555;margin-bottom:0.2rem;"
    "font-family:'Syne',sans-serif;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;}"
    ".note-comp{display:flex;align-items:flex-start;gap:0.75rem;background:#0f0f0f;"
    "border:1px solid #191919;border-radius:10px;padding:0.75rem 1rem;margin:0.4rem 0;}"
    ".note-comp-icon{color:#c8f04e;font-size:1rem;flex-shrink:0;}"
    ".note-comp-text{font-size:0.88rem;color:#d0cdc6;line-height:1.5;}"
    ".note-deep{background:#0f0f0f;border:1px solid #191919;border-radius:12px;"
    "padding:1.1rem 1.3rem;margin:0.6rem 0;}"
    ".note-deep-concept{font-family:'Syne',sans-serif;font-size:0.92rem;font-weight:800;"
    "color:#f0ede6;margin-bottom:0.3rem;}"
    ".note-deep-def{font-size:0.85rem;color:#c8f04e;margin-bottom:0.6rem;"
    "padding-left:0.8rem;border-left:2px solid #c8f04e;line-height:1.5;}"
    ".note-deep-point{display:flex;gap:0.6rem;font-size:0.85rem;color:#d0cdc6;"
    "line-height:1.5;margin:0.3rem 0;}"
    ".note-deep-dot{color:#555;flex-shrink:0;}"
    ".section-head{font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;"
    "letter-spacing:0.12em;color:#c8f04e;text-transform:uppercase;margin:0.8rem 0 0.5rem;}"
    ".empty-note{font-size:0.82rem;color:#333;padding:0.3rem 0;}"
    "</style>"
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "vectorstore": None, "full_text": "", "chunks": [],
    "chat_history": [], "pdf_audio": None,
    "last_audio_bytes": None, "voice_question": "", "auto_answer_pending": False,
    "summary_bullets": [], "quiz_items": [], "flowchart_code": "",
    "notes_level": None, "notes_data": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
# FIX 1 — audio_lang initialized OUTSIDE the loop so it's always set
if "audio_lang" not in st.session_state:
    st.session_state.audio_lang = "English"

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    name = "google/flan-t5-base"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSeq2SeqLM.from_pretrained(name)
    mdl.eval()
    return tok, mdl

# ── Core helpers ──────────────────────────────────────────────────────────────
def run_llm(prompt: str, max_new_tokens: int = 128) -> str:
    tok, mdl = load_llm()
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=max_new_tokens, num_beams=2, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True).strip()

def text_to_audio(text: str) -> bytes:
    lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
    selected_lang = st.session_state.get("audio_lang", "English")
    lang_code = lang_map.get(selected_lang, "en")
    if selected_lang != "English":
        try:
            text = GoogleTranslator(source="auto", target=lang_code).translate(text)
        except:
            pass
    MAX_CHARS = 4000
    chunks = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
    audio_bytes = b""
    for chunk in chunks:
        tts = gTTS(text=chunk, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = tmp.name
        tts.save(temp_path)
        with open(temp_path, "rb") as f:
            audio_bytes += f.read()
        os.unlink(temp_path)
    return audio_bytes

def transcribe(wav: bytes) -> str:
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(wav); p = f.name
    try:
        with sr.AudioFile(p) as src:
            return r.recognize_google(r.record(src))
    except:
        return ""
    finally:
        os.unlink(p)

# ── Feature functions ─────────────────────────────────────────────────────────

def answer_with_sources(question: str, vs) -> tuple:
    # Retrieve 6 chunks so multi-part answers (types, steps, lists) are complete
    docs = vs.similarity_search(question, k=6)
    ctx  = "\n\n".join(d.page_content for d in docs)
 
    q_lower = question.lower()
    is_list_q = any(w in q_lower for w in [
        "types", "kinds", "list", "what are", "name all", "enumerate",
        "steps", "methods", "techniques", "examples", "categories",
        "advantages", "disadvantages", "features", "components", "ways",
        "how many", "which are", "give all", "mention"
    ])
 
    if is_list_q:
        prompt = (
            "Using only the context below, list ALL relevant items for the question.\n"
            "Write each item on a new line starting with a dash (-).\n"
            "Be complete — include every item mentioned in the context.\n\n"
            "Context:\n" + ctx +
            "\n\nQuestion: " + question + "\nComplete list:"
        )

        ans = run_llm(prompt, max_new_tokens=250)

        parts = re.split(r"\d+\.\s*", ans)
        ans = "\n".join(dict.fromkeys(["- " + p.strip() for p in parts if len(p.strip()) > 3]))

	
    else:
        prompt = (
            "Answer the question fully and accurately using only the context below.\n\n"
            "Context:\n" + ctx +
            "\n\nQuestion: " + question + "\nAnswer:"
        )
        ans = run_llm(prompt, max_new_tokens=200)
 
    return ans, docs
def summarize_to_bullets(chunks: list) -> list:
    step   = max(1, len(chunks) // 6)
    sample = chunks[::step][:6]
    bullets, seen = [], set()
    for c in sample:
        point = run_llm(
            "In one complete sentence, what is the single most important fact or idea in this text?\n"
            "Text: " + c.page_content[:400] + "\nSentence:",
            max_new_tokens=60,
        ).strip().rstrip(".")
        if not point or len(point) < 15:
            continue
        key = point[:40].lower()
        if key not in seen:
            seen.add(key); bullets.append(point)
    return bullets[:8]

def generate_flowchart(chunks: list) -> str:
    step     = max(1, len(chunks) // 6)
    selected = chunks[::step][:6]
    nodes_text = []
    for c in selected:
        raw  = c.page_content.strip()
        sent = re.split(r'[.\n]', raw)[0].strip()
        if len(sent) < 8:
            sent = raw[:60].strip()
        sent = re.sub(r'[\[\]{}()"\'`<>|&]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()[:55]
        if sent:
            nodes_text.append(sent)
    if len(nodes_text) < 2:
        return ""
    ids, lines = "ABCDEFGHIJ", []
    for n, txt in enumerate(nodes_text):
        lines.append(f'    {ids[n]}["{txt}"]')
    for n in range(len(nodes_text) - 1):
        lines.append(f'    {ids[n]} --> {ids[n+1]}')
    return "flowchart TD\n" + "\n".join(lines)

def render_mermaid(code: str):
    safe = code.replace("`", "").replace("\\", "")
    html = (
        "<!DOCTYPE html><html><head>"
        "<script src='https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js'></script>"
        "<style>body{margin:0;background:#0d0d0d;}.wrap{background:#0f0f0f;border:1px solid #1e1e1e;"
        "border-radius:12px;padding:1.5rem;text-align:center;}.mermaid svg{max-width:100%;height:auto;}"
        "</style></head><body><div class='wrap'><div class='mermaid' id='chart'>" + safe + "</div></div>"
        "<script>mermaid.initialize({startOnLoad:false,theme:'dark',themeVariables:{"
        "primaryColor:'#1e1e1e',primaryTextColor:'#f0ede6',primaryBorderColor:'#c8f04e',"
        "lineColor:'#c8f04e',secondaryColor:'#141414',background:'#0d0d0d',"
        "nodeBorder:'#c8f04e',clusterBkg:'#111',titleColor:'#f0ede6',"
        "edgeLabelBackground:'#0d0d0d',fontSize:'15px'}});"
        "setTimeout(function(){mermaid.run({nodes:[document.getElementById('chart')]});},300);"
        "</script></body></html>"
    )
    components.html(html, height=480, scrolling=True)

def _short_distractor(chunk_text: str) -> str:
    sentences = [s.strip() for s in re.split(r'[.\n]', chunk_text) if len(s.strip()) > 10]
    if not sentences:
        return chunk_text[:50].strip()
    s = sentences[0][:60]
    if ' ' in s:
        s = s[:s.rfind(' ')]
    return s.strip()

def generate_quiz(chunks: list) -> list:
    import random as _r
    step     = max(1, len(chunks) // 5)
    selected = chunks[::step][:5]
    out      = []
    for idx, c in enumerate(selected):
        q = run_llm(
            "Write one clear, specific question about this text.\n"
            "Text: " + c.page_content[:500] + "\nQuestion:",
            max_new_tokens=50,
        ).strip().rstrip(".")
        if not q or len(q) < 8:
            continue
        if not q.endswith("?"):
            q += "?"
        a = run_llm(
            "Answer in one short phrase (max 10 words) using only this context.\n"
            "Context: " + c.page_content[:500] + "\nQuestion: " + q + "\nAnswer:",
            max_new_tokens=40,
        ).strip().rstrip(".")
        if not a or len(a) < 3:
            continue
        other = [ch for j, ch in enumerate(chunks) if j != idx]
        _r.shuffle(other)
        distractors = []
        for oc in other[:6]:
            d = _short_distractor(oc.page_content)
            if d and d.lower() not in a.lower() and d not in distractors and len(d) > 5:
                distractors.append(d)
            if len(distractors) == 3:
                break
        fillers = ["None of the above", "Not mentioned in the document", "Cannot be determined"]
        while len(distractors) < 3:
            distractors.append(fillers[len(distractors)])
        options = distractors[:3] + [a]
        _r.shuffle(options)
        out.append({"q": q, "a": a, "options": options, "correct": options.index(a)})
    return out

# ── Notes functions ───────────────────────────────────────────────────────────

# FIX 3 — Level 1: titled key facts (Topic | Fact format)
# ── Claude API helper for notes (gives much better quality than flan-t5) ─────
def _llm_notes(prompt: str, max_tokens: int = 800) -> str:
    """
    Use Groq (free) with Llama-3 for high quality notes generation.
    Get a free API key at: https://console.groq.com
    Falls back to flan-t5 if Groq key not set.
    """
    import os as _os

    # Read Groq API key
    api_key = _os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if not api_key:
        st.session_state["_claude_error"] = "No GROQ_API_KEY found — using flan-t5 fallback"
        return run_llm(prompt[:480], max_new_tokens=120)

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            },
            json={
                "model": "llama-3.1-8b-instant",
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        data = resp.json()
        st.session_state["_claude_error"] = str(data.get("error", "none"))

        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            st.session_state["_claude_error"] = "Groq error: " + msg
            return run_llm(prompt[:480], max_new_tokens=120)

        result = data["choices"][0]["message"]["content"].strip()
        st.session_state["_claude_error"] = "OK — got " + str(len(result)) + " chars"
        return result if result else run_llm(prompt[:480], max_new_tokens=120)

    except Exception as e:
        st.session_state["_claude_error"] = "Exception: " + str(e)
        return run_llm(prompt[:480], max_new_tokens=120)




# Level 1 — Quick Revision: titled one-liner cards
def generate_quick_revision(chunks: list) -> list:
    step   = max(1, len(chunks) // 10)
    sample = chunks[::step][:10]

    # Build a single combined prompt — one Claude call for all chunks
    combined = "\n\n---\n\n".join(
        "SECTION " + str(i+1) + ":\n" + c.page_content[:500]
        for i, c in enumerate(sample)
    )

    prompt = (
        "You are a study notes generator. Read the following sections from a document.\n"
        "For each section, produce exactly ONE flashcard in this format:\n"
        "TOPIC: [2-4 word topic name in CAPS]\n"
        "FACT: [One clear, complete sentence with the most important fact]\n\n"
        "Rules:\n"
        "- Every FACT must be a full sentence of at least 10 words\n"
        "- TOPIC must be specific (not generic words like 'Introduction' or 'Overview')\n"
        "- Do not repeat the same topic twice\n"
        "- Output ONLY the flashcards, no extra text\n\n"
        + combined
    )

    raw = _llm_notes(prompt, max_tokens=900)
    items, seen = [], set()

    # Strip markdown formatting Claude sometimes adds
    raw_clean = re.sub(r"[*_`]", "", raw)

    # Split on any line that starts with TOPIC (case-insensitive, with or without newline before)
    blocks = re.split(r"(?i)(?:^|\n)(?=TOPIC\s*:)", raw_clean.strip(), flags=re.MULTILINE)

    for block in blocks:
        t_match = re.search(r"(?i)TOPIC\s*:\s*(.+)", block)
        f_match = re.search(r"(?i)FACT\s*:\s*(.+(?:\n(?!TOPIC\s*:).+)*)", block, re.DOTALL)
        if not t_match:
            continue
        title = t_match.group(1).strip().title()
        if not f_match:
            continue
        fact = re.sub(r"\s+", " ", f_match.group(1)).strip().rstrip(".")
        if len(fact) < 8 or title.upper() in ("NONE", "N/A"):
            continue
        key = title[:30].lower()
        if key not in seen:
            seen.add(key)
            items.append({"title": title, "fact": fact})

    return items[:12]


# Level 2 — Study Notes: definitions, formulas, comparisons
def generate_study_notes(chunks: list) -> dict:
    """
    Ask Claude to return JSON directly — no regex parsing needed.
    Falls back gracefully if JSON is malformed.
    """
    step   = max(1, len(chunks) // 8)
    sample = chunks[::step][:8]

    combined = "\n\n".join(
        "[SECTION " + str(i+1) + "]\n" + c.page_content[:500]
        for i, c in enumerate(sample)
    )

    prompt = (
        "You are a study notes extractor. Read the document sections below.\n"
        "Return a JSON object with exactly these three keys: definitions, formulas, comparisons.\n\n"
        "definitions: array of objects with keys: term (string), definition (string, 2-3 sentences)\n"
        "formulas: array of objects with keys: name (string), value (string), explains (string, 1 sentence)\n"
        "comparisons: array of objects with keys: vs (string like 'A vs B'), difference (string, 2-3 sentences)\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON, no markdown, no explanation, no code fences\n"
        "- Each definition must be 2-3 complete sentences\n"
        "- Extract everything you can find\n"
        "- If nothing found for a key, use empty array []\n\n"
        "DOCUMENT:\n" + combined
    )

    raw = _llm_notes(prompt, max_tokens=1400)

    # strip any markdown code fences Claude might add
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()

    # store raw for debug
    st.session_state["_notes_raw"] = raw

    try:
        data = json.loads(raw)
        return {
            "definitions": data.get("definitions", []),
            "formulas":    data.get("formulas",    []),
            "comparisons": data.get("comparisons", []),
        }
    except Exception:
        # JSON parse failed — try to salvage partial data
        # by finding the largest valid JSON-like fragment
        try:
            # find first { and last }
            start = raw.index("{")
            end   = raw.rindex("}") + 1
            data  = json.loads(raw[start:end])
            return {
                "definitions": data.get("definitions", []),
                "formulas":    data.get("formulas",    []),
                "comparisons": data.get("comparisons", []),
            }
        except Exception:
            return {"definitions": [], "formulas": [], "comparisons": [], "_raw": raw}


# Level 3 — Deep Notes: concept card with definition + full explanation
def generate_deep_notes(chunks: list) -> list:
    step   = max(1, len(chunks) // 6)
    sample = chunks[::step][:6]

    combined = "\n\n---\n\n".join(
        "SECTION " + str(i+1) + ":\n" + c.page_content[:700]
        for i, c in enumerate(sample)
    )

    prompt = (
        "You are a study companion writing detailed notes for a student.\n"
        "Read the document sections and for each section create a concept note card.\n\n"
        "For each section, output a note card in this exact format:\n\n"
        "CONCEPT: [Name of the main concept/topic in 2-5 words]\n"
        "DEFINITION: [A clear, complete definition in 2-3 sentences. Must be thorough enough for a student to understand without reading the original text.]\n"
        "EXPLANATION:\n"
        "- [Point 1: explain one important aspect, at least 1-2 full sentences]\n"
        "- [Point 2: explain another aspect, at least 1-2 full sentences]\n"
        "- [Point 3: give a practical example or application, at least 1 full sentence]\n"
        "REMEMBER: [One memorable sentence that helps the student retain this concept]\n\n"
        "Rules:\n"
        "- Each DEFINITION must be 2-3 complete sentences minimum\n"
        "- Each bullet point must be at least one complete sentence\n"
        "- Write in simple, clear language a student can understand\n"
        "- Do not skip any section — produce one note card per section\n"
        "- Do not repeat concepts across cards\n\n"
        + combined
    )

    raw = _llm_notes(prompt, max_tokens=1500)
    notes, seen = [], set()

    # Strip markdown that Claude sometimes adds
    raw_clean = re.sub(r"[*_`#]", "", raw)

    # Split into blocks on any line starting with CONCEPT (case-insensitive)
    card_blocks = re.split(
        r"(?i)(?:^|\n)(?=CONCEPT\s*:)",
        raw_clean.strip(),
        flags=re.MULTILINE
    )

    for block in card_blocks:
        c_match = re.search(r"(?i)CONCEPT\s*:\s*(.+)", block)
        if not c_match:
            continue
        concept = c_match.group(1).strip().title()
        if not concept or len(concept) < 2:
            continue

        # Definition — everything between DEFINITION: and EXPLANATION: or REMEMBER:
        d_match = re.search(
            r"(?i)DEFINITION\s*:\s*(.+?)(?=\n(?:EXPLANATION|REMEMBER)\s*:|$)",
            block, re.DOTALL
        )
        definition = re.sub(r"\s+", " ", d_match.group(1)).strip() if d_match else ""

        # Remember
        r_match  = re.search(r"(?i)REMEMBER\s*:\s*(.+)", block)
        remember = r_match.group(1).strip() if r_match else ""

        # Explanation bullet points
        e_match = re.search(
            r"(?i)EXPLANATION\s*:\s*\n?(.*?)(?=\nREMEMBER\s*:|$)",
            block, re.DOTALL
        )
        points = []
        if e_match:
            for line in e_match.group(1).split("\n"):
                pt = line.strip().lstrip("-•·▸* 1234567890.)").strip()
                if pt and len(pt) > 10:
                    points.append(pt)

        key = concept[:30].lower()
        if key not in seen:
            seen.add(key)
            notes.append({
                "concept":    concept,
                "definition": definition if definition else "See explanation below.",
                "points":     points[:4],
                "remember":   remember,
            })

    return notes[:8]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:0.2rem 0 1.4rem 0;'>"
        "<p style='font-family:Syne,sans-serif;font-weight:800;font-size:1.35rem;"
        "color:#f0ede6;margin:0;'>📄 AskMyDoc</p>"
        "<p style='font-size:0.75rem;color:#444;margin:0.25rem 0 0 0;letter-spacing:0.04em;'>"
        "AI PDF Study Companion</p></div>",
        unsafe_allow_html=True,
    )

    # FIX 1 — Language selector with visible label
    st.markdown("<span class='sidebar-label'>🌐 Audio Language</span>", unsafe_allow_html=True)
    st.selectbox("", ["English", "Hindi", "Marathi"], key="audio_lang", label_visibility="collapsed")

    st.markdown("<span class='sidebar-label'>Upload PDF</span>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded is not None and st.session_state.vectorstore is None:
        with st.spinner("Indexing…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read()); path = tmp.name
            docs = PyPDFLoader(path).load()
            st.session_state.full_text = "\n\n".join(d.page_content for d in docs)
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
            st.session_state.chunks      = chunks
            st.session_state.vectorstore = FAISS.from_documents(chunks, load_embeddings())
            for k in ["summary_bullets", "quiz_items", "chat_history"]:
                st.session_state[k] = []
            for k in ["flowchart_code", "voice_question"]:
                st.session_state[k] = ""
            st.session_state.pdf_audio  = None
            st.session_state.notes_level = None
            st.session_state.notes_data  = None
            os.unlink(path)
        st.success("✓ Ready!")

    elif uploaded is None and st.session_state.vectorstore is not None:
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    if st.session_state.vectorstore:
        pages      = len(set(c.metadata.get("page", 0) for c in st.session_state.chunks))
        chunks_len = len(st.session_state.chunks)
        st.markdown(
            "<div class='stat-row'>"
            "<div class='stat-pill'><strong>" + str(pages) + "</strong>Pages</div>"
            "<div class='stat-pill'><strong>" + str(chunks_len) + "</strong>Chunks</div>"
            "</div>", unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<span class='sidebar-label'>🔊 Read PDF Aloud</span>", unsafe_allow_html=True)
        if st.button("Generate Audio"):
            with st.spinner("Converting to speech…"):
                st.session_state.pdf_audio = text_to_audio(st.session_state.full_text)
        if st.session_state.pdf_audio:
            st.audio(st.session_state.pdf_audio, format="audio/mp3")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<span class='sidebar-label'>🎙️ Voice Question</span>", unsafe_allow_html=True)
        st.markdown("<p class='mic-hint'>Click mic → speak → stop → go to Chat tab.</p>", unsafe_allow_html=True)
        rec = audio_recorder(
            text="", recording_color="#c8f04e", neutral_color="#333",
            icon_name="microphone", icon_size="2x",
            pause_threshold=3.0, sample_rate=16000,
        )
        if rec and rec != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = rec
            with st.spinner("Transcribing…"):
                t = transcribe(rec)
            if t:
                st.session_state.voice_question      = t
                st.session_state.auto_answer_pending = True
                st.success("🎤 Heard: " + t + " — answering…")
                st.rerun()
            else:
                st.warning("Couldn't hear clearly. Try again.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<p class='page-title'>AskMyDoc <span class='accent'>·</span> AI-powered PDF study companion</p>"
    "<p class='page-sub'>Upload a PDF in the sidebar — then Chat, Summarize, Visualize, take Notes or Quiz yourself.</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Welcome screen ─────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    # FIX 2 — Notes card added to welcome screen, 3-column grid
    st.markdown(
        "<div class='welcome-card'>"
        "<div style='font-size:3rem;margin-bottom:0.8rem;'>📄</div>"
        "<h3 style='font-family:Syne,sans-serif;color:#f0ede6;font-size:1.3rem;margin:0 0 0.4rem 0;'>"
        "Upload a PDF to unlock all features</h3>"
        "<p style='color:#444;font-size:0.87rem;margin:0 0 1.5rem 0;'>Use the sidebar on the left to get started.</p>"
        "<div class='feature-grid'>"
        "<div class='feature-item'><span>💬</span><strong>Chat Q&A</strong>Ask anything with source highlighting</div>"
        "<div class='feature-item'><span>📋</span><strong>Smart Summary</strong>Key bullet points + audio playback</div>"
        "<div class='feature-item'><span>🔀</span><strong>Flowchart</strong>Visual concept map of your document</div>"
        "<div class='feature-item'><span>📒</span><strong>Smart Notes</strong>Quick Revision, Study Notes &amp; Deep Notes</div>"
        "<div class='feature-item'><span>📝</span><strong>MCQ Quiz</strong>Auto-generated questions to test yourself</div>"
        "<div class='feature-item'><span>🌐</span><strong>Multilingual</strong>Audio in English, Hindi or Marathi</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

# ── Main tabs ─────────────────────────────────────────────────────────────────
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["💬  Chat", "📋  Summary", "🔀  Flowchart", "📒  Notes", "📝  Quiz"]
    )

    # ══ TAB 1 — CHAT Q&A ══════════════════════════════════════════════════════
    with tab1:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>Chat Q&A </div>"
            "<p class='info-card-text'>Ask questions by typing or use the mic in the sidebar. "
            "</p></div>",
            unsafe_allow_html=True,
        )
        for msg in st.session_state.chat_history:
            st.markdown(
                "<div class='chat-bubble-user'><div class='chat-role chat-role-user'>You</div>"
                "<div class='chat-text'>" + msg["question"] + "</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='chat-bubble-bot'><div class='chat-role chat-role-bot'>AskMyDoc</div>"
                "<div class='chat-text'>" + msg["answer"] + "</div></div>",
                unsafe_allow_html=True,
            )
           
        if st.session_state.chat_history:
            last_ans = st.session_state.chat_history[-1]["answer"]
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            cola, colb = st.columns([3, 1])
            with cola:
                if st.button("🔊 Play Latest Answer"):
                    with st.spinner("Generating audio…"):
                        st.session_state["last_ans_audio"] = text_to_audio(last_ans)
            with colb:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            if "last_ans_audio" in st.session_state and st.session_state["last_ans_audio"]:
                st.audio(st.session_state["last_ans_audio"], format="audio/mp3", autoplay=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.session_state.auto_answer_pending and st.session_state.voice_question.strip():
            auto_q = st.session_state.voice_question.strip()
            st.session_state.auto_answer_pending = False
            st.session_state.voice_question      = ""
            with st.spinner("Finding answer…"):
                ans, srcs = answer_with_sources(auto_q, st.session_state.vectorstore)
            with st.spinner("Generating audio answer…"):
                auto_audio = text_to_audio(ans)
            st.session_state.chat_history.append({"question": auto_q, "answer": ans})
            st.session_state["last_ans_audio"] = auto_audio
            st.rerun()

        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "", placeholder="Type your question… or use the 🎙️ mic in the sidebar",
                label_visibility="collapsed", value=st.session_state.voice_question, key="qinput",
            )
        with col2:
            ask = st.button("Ask →")
        if ask and question.strip():
            with st.spinner("Thinking…"):
                ans, __ = answer_with_sources(question.strip(), st.session_state.vectorstore)
            st.session_state.chat_history.append({"question": question.strip(), "answer": ans, "sources": srcs})
            st.session_state.voice_question    = ""
            st.session_state["last_ans_audio"] = None
            st.rerun()

    # ══ TAB 2 — SUMMARY ═══════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>Smart Summary</div>"
            "<p class='info-card-text'>Condenses your PDF into concise bullet points using AI summarization.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("✨ Generate Summary"):
            with st.spinner("Summarizing… may take ~30s for large PDFs."):
                st.session_state.summary_bullets = summarize_to_bullets(st.session_state.chunks)
            st.rerun()
        if st.session_state.summary_bullets:
            for point in st.session_state.summary_bullets:
                st.markdown(
                    "<div class='bullet-item'><span class='bullet-dot'>▸</span>"
                    "<span class='bullet-text'>" + point + "</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
            if st.button("🔊 Listen to Summary"):
                with st.spinner("Generating audio…"):
                    s_audio = text_to_audio(". ".join(st.session_state.summary_bullets))
                st.audio(s_audio, format="audio/mp3")

    # ══ TAB 3 — FLOWCHART ═════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>Concept Flowchart</div>"
            "<p class='info-card-text'>Extracts the main topics from your document and renders an interactive flowchart.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("🔀 Generate Flowchart"):
            with st.spinner("Extracting concepts…"):
                st.session_state.flowchart_code  = generate_flowchart(st.session_state.chunks)
                st.session_state["_flow_tried"] = True
        if st.session_state.flowchart_code:
            render_mermaid(st.session_state.flowchart_code)
            with st.expander("View Mermaid diagram code"):
                st.code(st.session_state.flowchart_code, language="text")
        elif not st.session_state.flowchart_code and st.session_state.get("_flow_tried"):
            st.warning("Could not build a flowchart. Try a more structured document.")

    # ══ TAB 4 — NOTES ═════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>Smart Notes</div>"
            "<p class='info-card-text'>Generate structured study notes from your PDF. "
            "Choose a note style and click Generate.</p></div>",
            unsafe_allow_html=True,
        )

        for _k, _v in [("notes_level", None), ("notes_data", None)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        # ── Level selector cards ──────────────────────────────────────────────
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            st.markdown(
                "<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:12px;"
                "padding:1rem;text-align:center;'>"
                "<div style='font-size:1.5rem;'>⚡</div>"
                "<div style='font-family:Syne,sans-serif;font-size:0.82rem;font-weight:700;"
                "color:#f0ede6;margin:0.3rem 0 0.2rem;'>Quick Revision</div>"
                "<div style='font-size:0.75rem;color:#555;'>Titled one-line facts<br>for last-minute revision</div>"
                "</div>", unsafe_allow_html=True,
            )
            if st.button("Generate ⚡", key="btn_quick"):
                st.session_state.notes_level = "quick"
                st.session_state.notes_data  = None
                st.rerun()
        with nc2:
            st.markdown(
                "<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:12px;"
                "padding:1rem;text-align:center;'>"
                "<div style='font-size:1.5rem;'>📘</div>"
                "<div style='font-family:Syne,sans-serif;font-size:0.82rem;font-weight:700;"
                "color:#f0ede6;margin:0.3rem 0 0.2rem;'>Study Notes</div>"
                "<div style='font-size:0.75rem;color:#555;'>Definitions, formulas<br>&amp; comparisons</div>"
                "</div>", unsafe_allow_html=True,
            )
            if st.button("Generate 📘", key="btn_study"):
                st.session_state.notes_level = "study"
                st.session_state.notes_data  = None
                st.rerun()
        with nc3:
            st.markdown(
                "<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:12px;"
                "padding:1rem;text-align:center;'>"
                "<div style='font-size:1.5rem;'>🧠</div>"
                "<div style='font-family:Syne,sans-serif;font-size:0.82rem;font-weight:700;"
                "color:#f0ede6;margin:0.3rem 0 0.2rem;'>Deep Notes</div>"
                "<div style='font-size:0.75rem;color:#555;'>Concept + definition<br>+ key explanation points</div>"
                "</div>", unsafe_allow_html=True,
            )
            if st.button("Generate 🧠", key="btn_deep"):
                st.session_state.notes_level = "deep"
                st.session_state.notes_data  = None
                st.rerun()

        # ── Generate when level chosen ────────────────────────────────────────
        if st.session_state.notes_level and st.session_state.notes_data is None:
            lmap = {"quick": "Quick Revision", "study": "Study Notes", "deep": "Deep Notes"}
            with st.spinner("Generating " + lmap[st.session_state.notes_level] + "… (~30–60s)"):
                if st.session_state.notes_level == "quick":
                    st.session_state.notes_data = generate_quick_revision(st.session_state.chunks)
                elif st.session_state.notes_level == "study":
                    st.session_state.notes_data = generate_study_notes(st.session_state.chunks)
                else:
                    st.session_state.notes_data = generate_deep_notes(st.session_state.chunks)
            st.rerun()

        # ── Render notes ──────────────────────────────────────────────────────
        if st.session_state.notes_data is not None:
            level = st.session_state.notes_level
            data  = st.session_state.notes_data
            st.markdown("<hr>", unsafe_allow_html=True)

                                                     # ── QUICK REVISION ────────────────────────────────────────────────
            if level == "quick":

                st.markdown(
                    "<p class='section-head'>⚡ Quick Revision Facts</p>",
                    unsafe_allow_html=True
                )

                cleaned_data = []

                # normal model output
                if isinstance(data, list) and len(data) > 0:

                    for item in data:

                        if isinstance(item, dict):

                            title = str(
                                item.get("title")
                                or item.get("topic")
                                or item.get("heading")
                                or item.get("concept")
                                or "Key Concept"
                            ).strip()

                            fact = str(
                                item.get("fact")
                                or item.get("point")
                                or item.get("statement")
                                or item.get("content")
                                or item.get("text")
                                or ""
                            ).strip()

                            if len(title) > 2 and len(fact) > 20:

                                cleaned_data.append({
                                    "title": title,
                                    "fact": fact
                                })


                # fallback if model output is empty or poor
                if not cleaned_data:

                    fallback_points = []

                    # use summary bullets if available
                    if st.session_state.summary_bullets:

                        for s in st.session_state.summary_bullets:

                            if len(s) > 40 and len(s.split()) > 6:

                                fallback_points.append(s)


                    # otherwise extract conceptual sentences from chunks
                    else:

                        for c in st.session_state.chunks[:30]:

                            text = c.page_content.strip()

                            sentences = re.split(r"[.]", text)

                            for s in sentences:

                                s = s.strip()
                                s_lower = s.lower()

                                if (
                                    len(s) > 45
                                    and len(s.split()) > 7
                                    and not s.startswith(("•", "-", "*"))
                                    and not s_lower.startswith(("example", "for example"))
                                    and not s.endswith(":")
                                    and (
                                        " is " in s_lower
                                        or " refers to " in s_lower
                                        or " defined as " in s_lower
                                        or " used to " in s_lower
                                        or " means " in s_lower
                                    )
                                ):

                                    fallback_points.append(s)

                                if len(fallback_points) >= 8:
                                    break

                            if len(fallback_points) >= 8:
                                break


                    for point in fallback_points[:8]:

                        cleaned_data.append({
                            "title": "Key Concept",
                            "fact": point.strip()
                        })


                # display results
                if cleaned_data:

                    dl_text = "QUICK REVISION NOTES\n" + "=" * 40 + "\n\n"

                    for i, item in enumerate(cleaned_data):

                        st.markdown(
                            "<div class='note-qr'>"
                            "<div class='note-qr-num'>" + str(i+1) + "</div>"
                            "<div class='note-qr-body'>"
                            "<div class='note-qr-title'>" + item["title"] + "</div>"
                            "<div class='note-qr-fact'>" + item["fact"] + "</div>"
                            "</div></div>",
                            unsafe_allow_html=True
                        )

                        dl_text += (
                            str(i+1)
                            + ". ["
                            + item["title"]
                            + "]\n   "
                            + item["fact"]
                            + "\n\n"
                        )


                    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:

                        st.download_button(
                            "⬇️ Download (.txt)",
                            data=dl_text,
                            file_name="quick_revision.txt",
                            mime="text/plain"
                        )


                    with col2:

                        if st.button("🔊 Listen", key="aud_quick"):

                            spoken = ". ".join(
                                item["title"] + ": " + item["fact"]
                                for item in cleaned_data
                            )

                            with st.spinner("Generating audio…"):

                                audio_bytes = text_to_audio(spoken)

                                st.audio(audio_bytes, format="audio/mp3")


                else:

                    st.warning(
                        "Could not extract revision points from this PDF."
                    )

            # ── STUDY NOTES ───────────────────────────────────────────────────
            elif level == "study":
                dl_text = "STUDY NOTES\n" + "="*40 + "\n\n"

          

                # Definitions
                st.markdown("<p class='section-head'>📖 Definitions</p>", unsafe_allow_html=True)
                if data["definitions"]:
                    dl_text += "DEFINITIONS\n" + "-"*30 + "\n"
                    for item in data["definitions"]:
                        st.markdown(
                            "<div class='note-def'>"
                            "<div class='note-def-term'>" + item["term"] + "</div>"
                            "<div class='note-def-text'>" + item["definition"] + "</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        dl_text += "• " + item["term"] + ": " + item["definition"] + "\n"
                else:
                    st.markdown("<p class='empty-note'>No definitions found in this document.</p>", unsafe_allow_html=True)

                # Formulas
                st.markdown("<p class='section-head'>🔢 Formulas &amp; Key Values</p>", unsafe_allow_html=True)
                if data["formulas"]:
                    dl_text += "\nFORMULAS & KEY VALUES\n" + "-"*30 + "\n"
                    for item in data["formulas"]:
                        explains_html = (
                            "<div style='font-size:0.78rem;color:#888;margin-top:0.35rem;'>"
                            + item.get("explains","") + "</div>"
                        ) if item.get("explains") else ""
                        st.markdown(
                            "<div class='note-formula'>"
                            "<div class='note-formula-label'>" + item["name"] + "</div>"
                            + item["value"]
                            + explains_html +
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        dl_text += "• " + item["name"] + ": " + item["value"] + (
                            "\n  → " + item["explains"] if item.get("explains") else ""
                        ) + "\n"
                else:
                    st.markdown("<p class='empty-note'>No formulas or key values found in this document.</p>", unsafe_allow_html=True)

                # Comparisons
                st.markdown("<p class='section-head'>⚖️ Comparisons</p>", unsafe_allow_html=True)
                if data["comparisons"]:
                    dl_text += "\nCOMPARISONS\n" + "-"*30 + "\n"
                    for item in data["comparisons"]:
                        # Support both old string format and new dict format
                        if isinstance(item, dict):
                            vs_txt   = item.get("vs", "")
                            diff_txt = item.get("difference", "")
                        else:
                            vs_txt   = item
                            diff_txt = ""
                        st.markdown(
                            "<div class='note-comp'>"
                            "<div style='width:100%;'>"
                            "<div style='font-family:Syne,sans-serif;font-size:0.82rem;"
                            "font-weight:700;color:#c8f04e;margin-bottom:0.35rem;'>"
                            + vs_txt + "</div>"
                            + ("<div style='font-size:0.86rem;color:#d0cdc6;line-height:1.55;'>"
                               + diff_txt + "</div>" if diff_txt else "")
                            + "</div></div>",
                            unsafe_allow_html=True,
                        )
                        dl_text += "• " + vs_txt + (
                            "\n  " + diff_txt if diff_txt else ""
                        ) + "\n"
                else:
                    st.markdown("<p class='empty-note'>No comparisons found in this document.</p>", unsafe_allow_html=True)

                st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
                all_spoken = (
                    "Definitions. " +
                    ". ".join(i["term"] + " means " + i["definition"] for i in data["definitions"]) +
                    " Formulas. " +
                    ". ".join(
    				c["vs"] + ". " + c["difference"]
    				for c in data["comparisons"]
			)
                )
                ca, cb = st.columns(2)
                with ca:
                    st.download_button("⬇️ Download (.txt)", data=dl_text,
                                       file_name="study_notes.txt", mime="text/plain")
                with cb:
                    if st.button("🔊 Listen", key="aud_study"):
                        with st.spinner("Generating audio…"):
                            st.audio(text_to_audio(all_spoken), format="audio/mp3")

            # ── DEEP NOTES ────────────────────────────────────────────────────
            elif level == "deep":
                st.markdown("<p class='section-head'>🧠 Deep Notes</p>", unsafe_allow_html=True)
                if data:
                    dl_text = "DEEP NOTES\n" + "="*40 + "\n\n"
                    for i, note in enumerate(data):
                        # Build points HTML
                        pts_html = "".join(
                            "<div class='note-deep-point'>"
                            "<span class='note-deep-dot'>▸</span>"
                            "<span>" + pt + "</span></div>"
                            for pt in note["points"]
                        ) if note["points"] else ""

                        remember_html = (
                            "<div style='background:#0a0f00;border-left:2px solid #555;"
                            "border-radius:0 6px 6px 0;padding:0.5rem 0.8rem;margin-top:0.6rem;"
                            "font-size:0.8rem;color:#888;font-style:italic;'>"
                            "💡 " + note.get("remember","") + "</div>"
                        ) if note.get("remember") else ""

                        st.markdown(
                            "<div class='note-deep'>"
                            "<div class='note-deep-concept'>🔷 " + note["concept"] + "</div>"
                            "<div class='note-deep-def'>" + note["definition"] + "</div>"
                            + pts_html
                            + remember_html +
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        dl_text += (
                            "CONCEPT: " + note["concept"] + "\n"
                            "Definition: " + note["definition"] + "\n"
                            + "\n".join("• " + pt for pt in note["points"])
                            + ("\nRemember: " + note["remember"] if note.get("remember") else "")
                            + "\n\n"
                        )
                    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
                    spoken_deep = ". ".join(
                        note["concept"] + ". " + note["definition"] + ". " +
                        " ".join(note["points"])
                        for note in data
                    )
                    ca, cb = st.columns(2)
                    with ca:
                        st.download_button("⬇️ Download (.txt)", data=dl_text,
                                           file_name="deep_notes.txt", mime="text/plain")
                    with cb:
                        if st.button("🔊 Listen", key="aud_deep"):
                            with st.spinner("Generating audio…"):
                                st.audio(text_to_audio(spoken_deep), format="audio/mp3")
                else:
                    st.info("Could not generate deep notes. Try a more structured PDF.")

    # ══ TAB 5 — QUIZ ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>MCQ Quiz</div>"
            "<p class='info-card-text'>Auto-generated multiple choice questions from your document. "
            "Pick an option for each question, then Submit to see your score.</p></div>",
            unsafe_allow_html=True,
        )
        for _k, _v in [("quiz_answers", {}), ("quiz_result", None), ("_quiz_tried", False)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        col_gen, col_reset = st.columns([2, 1])
        with col_gen:
            if st.button("📝 Generate Quiz"):
                with st.spinner("Generating MCQ questions… (~30–45s)"):
                    st.session_state.quiz_items      = generate_quiz(st.session_state.chunks)
                    st.session_state["_quiz_tried"]  = True
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_result"]  = None
                st.rerun()
        with col_reset:
            if st.session_state.quiz_items and st.button("🔄 New Quiz"):
                st.session_state.quiz_items      = []
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_result"]  = None
                st.rerun()

        if st.session_state.quiz_items:
            disabled = bool(st.session_state["quiz_result"])
            for i, item in enumerate(st.session_state.quiz_items):
                st.markdown(
                    "<div class='quiz-card'>"
                    "<div class='quiz-num'>Question " + str(i+1) + " of "
                    + str(len(st.session_state.quiz_items)) + "</div>"
                    "<div class='quiz-q'>❓ " + item["q"] + "</div></div>",
                    unsafe_allow_html=True,
                )
                opts   = ["— select an option —"] + item["options"]
                chosen = st.radio("Options", opts, key="mcq_" + str(i),
                                  label_visibility="collapsed", disabled=disabled)
                st.session_state["quiz_answers"][str(i)] = chosen
                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            if not disabled:
                if st.button("✅ Submit & See Results", type="primary"):
                    score, results = 0, []
                    for i, item in enumerate(st.session_state.quiz_items):
                        chosen = st.session_state["quiz_answers"].get(str(i), "")
                        ok     = (chosen == item["options"][item["correct"]])
                        if ok: score += 1
                        results.append({"q": item["q"], "chosen": chosen,
                                        "correct": item["options"][item["correct"]], "ok": ok})
                    st.session_state["quiz_result"] = {
                        "score": score, "total": len(st.session_state.quiz_items), "details": results
                    }
                    st.rerun()

            if st.session_state["quiz_result"]:
                r     = st.session_state["quiz_result"]
                pct   = int((r["score"] / r["total"]) * 100)
                emoji = "🏆" if pct >= 80 else ("👍" if pct >= 50 else "📖")
                color = "#c8f04e" if pct >= 80 else ("#f0a050" if pct >= 50 else "#e05050")
                msg   = "Excellent!" if pct >= 80 else ("Good effort!" if pct >= 50 else "Keep reading!")
                st.markdown(
                    "<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:14px;"
                    "padding:1.8rem;text-align:center;margin:1rem 0;'>"
                    "<div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;color:"
                    + color + ";line-height:1;'>" + str(pct) + "%</div>"
                    "<div style='font-size:1rem;color:" + color + ";font-family:Syne,sans-serif;"
                    "font-weight:700;margin-top:0.4rem;'>" + emoji + "  " + msg + "</div>"
                    "<div style='font-size:0.82rem;color:#555;margin-top:0.3rem;'>"
                    + str(r["score"]) + " correct out of " + str(r["total"]) + " questions</div>"
                    "</div>", unsafe_allow_html=True,
                )
                st.markdown(
                    "<p style='font-family:Syne,sans-serif;font-size:0.68rem;font-weight:700;"
                    "letter-spacing:0.12em;color:#555;text-transform:uppercase;margin:0.5rem 0;'>"
                    "Answer Review</p>", unsafe_allow_html=True,
                )
                for d in r["details"]:
                    border     = "#1e2d04" if d["ok"] else "#2d0e0e"
                    icon       = "✅" if d["ok"] else "❌"
                    your_color = "#c8f04e" if d["ok"] else "#e05050"
                    st.markdown(
                        "<div style='background:#0a0a0a;border:1px solid " + border + ";"
                        "border-radius:10px;padding:1rem 1.2rem;margin:0.45rem 0;'>"
                        "<div style='font-size:0.88rem;color:#f0ede6;font-weight:500;"
                        "margin-bottom:0.6rem;'>" + icon + " " + d["q"] + "</div>"
                        "<div style='font-size:0.82rem;color:#666;margin-bottom:0.25rem;'>"
                        "Your answer: <span style='color:" + your_color + ";font-weight:500;'>"
                        + (d["chosen"] if d["chosen"] and d["chosen"] != "— select an option —"
                           else "<em>no answer selected</em>") + "</span></div>"
                        + ("" if d["ok"] else
                           "<div style='font-size:0.82rem;color:#666;'>"
                           "Correct answer: <span style='color:#c8f04e;font-weight:500;'>"
                           + d["correct"] + "</span></div>")
                        + "</div>", unsafe_allow_html=True,
                    )

        elif not st.session_state.quiz_items and st.session_state.get("_quiz_tried"):
            st.info("Quiz generation failed. Try a longer or more structured document.")
