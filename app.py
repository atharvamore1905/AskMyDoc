import streamlit as st
import tempfile, os, io, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import streamlit.components.v1 as components
from gtts import gTTS
import tempfile

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
    ".feature-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:0.8rem;margin-top:1.5rem;}"
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
    "</style>"
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "vectorstore": None, "full_text": "", "chunks": [],
    "chat_history": [], "pdf_audio": None,
    "last_audio_bytes": None, "voice_question": "", "auto_answer_pending": False,
    "summary_bullets": [], "quiz_items": [], "flowchart_code": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
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

    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Marathi": "mr"
    }

    lang_code = lang_map.get(st.session_state.audio_lang, "en")

    MAX_CHARS = 4000

    parts = [
        text[i:i+MAX_CHARS]
        for i in range(0, len(text), MAX_CHARS)
    ]

    audio_bytes = b""

    for p in parts:

        tts = gTTS(
            text=p,
            lang=lang_code,
            slow=False
        )

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

# 1. RAG answer with source chunks returned
def answer_with_sources(question: str, vs) -> tuple:
    docs = vs.similarity_search(question, k=3)
    ctx  = "\n\n".join(d.page_content for d in docs)
    ans  = run_llm(
        "Answer concisely using only the context.\n\nContext:\n" + ctx +
        "\n\nQuestion: " + question + "\nAnswer:",
        max_new_tokens=128,
    )
    return ans, docs

# 2. Map-reduce summarization to bullet points
def summarize_to_bullets(chunks: list) -> list:
    """Extract one key point per chunk, deduplicate, return best 8."""
    step    = max(1, len(chunks) // 12)          # sample up to 12 evenly spaced chunks
    sample  = chunks[::step][:12]
    bullets = []
    seen    = set()
    for c in sample:
        point = run_llm(
            "In one complete sentence, what is the single most important fact or idea in this text?\n"
            "Text: " + c.page_content[:600] + "\nSentence:",
            max_new_tokens=60,
        ).strip().rstrip(".")
        if not point or len(point) < 15:
            continue
        key = point[:40].lower()
        if key not in seen:
            seen.add(key)
            bullets.append(point)
    return bullets[:8]

# 3. Flowchart generation (Mermaid)
def generate_flowchart(chunks: list) -> str:
    """
    Build a flowchart from the document without relying on the LLM.
    Strategy: take the first complete sentence from 6 evenly-spaced chunks
    as the 'topic' of that section, then wire them top-to-bottom.
    This is deterministic and never produces broken Mermaid syntax.
    """
    step     = max(1, len(chunks) // 6)
    selected = chunks[::step][:6]
    nodes_text = []
    for c in selected:
        # grab first sentence (split on . or newline)
        raw = c.page_content.strip()
        # take first sentence up to 60 chars
        sent = re.split(r'[.\n]', raw)[0].strip()
        if len(sent) < 8:
            # fallback: first 60 chars of chunk
            sent = raw[:60].strip()
        # sanitize for mermaid: remove all special chars except spaces
        sent = re.sub(r'[\[\]{}()"\'`<>|&]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()[:55]
        if sent:
            nodes_text.append(sent)

    if len(nodes_text) < 2:
        return ""

    ids   = "ABCDEFGHIJ"
    lines = []
    # nodes
    for n, txt in enumerate(nodes_text):
        lines.append(f'    {ids[n]}["{txt}"]')
    # edges
    for n in range(len(nodes_text) - 1):
        lines.append(f'    {ids[n]} --> {ids[n+1]}')

    return "flowchart TD\n" + "\n".join(lines)

def render_mermaid(code: str):
    safe = code.replace("`", "").replace("\\", "")
    html = """
<!DOCTYPE html><html><head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
<style>
  body{margin:0;background:#0d0d0d;}
  .wrap{background:#0f0f0f;border:1px solid #1e1e1e;border-radius:12px;
        padding:1.5rem;text-align:center;}
  .mermaid svg{max-width:100%;height:auto;}
</style></head><body>
<div class="wrap">
  <div class="mermaid" id="chart">""" + safe + """</div>
</div>
<script>
  mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    themeVariables: {
      primaryColor: "#1e1e1e",
      primaryTextColor: "#f0ede6",
      primaryBorderColor: "#c8f04e",
      lineColor: "#c8f04e",
      secondaryColor: "#141414",
      background: "#0d0d0d",
      nodeBorder: "#c8f04e",
      clusterBkg: "#111",
      titleColor: "#f0ede6",
      edgeLabelBackground: "#0d0d0d",
      fontSize: "15px"
    }
  });
  setTimeout(function(){
    mermaid.run({ nodes: [document.getElementById("chart")] });
  }, 300);
</script>
</body></html>"""
    components.html(html, height=480, scrolling=True)

# 4. Quiz generator
def _short_distractor(chunk_text: str) -> str:
    """Pull a short phrase from a chunk to use as a wrong option."""
    sentences = [s.strip() for s in re.split(r'[.\n]', chunk_text) if len(s.strip()) > 10]
    if not sentences:
        return chunk_text[:50].strip()
    # pick a sentence and trim to ≤60 chars at a word boundary
    s = sentences[0][:60]
    if ' ' in s:
        s = s[:s.rfind(' ')]
    return s.strip()

def generate_quiz(chunks: list) -> list:
    """Generate MCQ questions: 1 correct answer + 3 distractors from other chunks."""
    step     = max(1, len(chunks) // 5)
    selected = chunks[::step][:5]
    out      = []
    all_chunks = chunks  # pool for distractors

    for idx, c in enumerate(selected):
        q = run_llm(
            "Write one clear, specific question about this text.\n"
            "Text: " + c.page_content[:500] + "\nQuestion:",
            max_new_tokens=50,
        ).strip().rstrip(".")
        if not q or len(q) < 8 or "?" not in q and len(q) < 15:
            q = q + "?" if q and not q.endswith("?") else q
        a = run_llm(
            "Answer in one short phrase (max 10 words) using only this context.\n"
            "Context: " + c.page_content[:500] + "\nQuestion: " + q + "\nAnswer:",
            max_new_tokens=40,
        ).strip().rstrip(".")

        if not q or not a or len(a) < 3:
            continue

        # Build 3 distractors from OTHER chunks
        other = [ch for j, ch in enumerate(all_chunks) if j != idx]
        import random as _r
        _r.shuffle(other)
        distractors = []
        for oc in other[:6]:
            d = _short_distractor(oc.page_content)
            if d and d.lower() not in a.lower() and d not in distractors and len(d) > 5:
                distractors.append(d)
            if len(distractors) == 3:
                break
        # pad with generic fillers if not enough distractors
        fillers = ["None of the above", "Not mentioned in the document", "Cannot be determined"]
        while len(distractors) < 3:
            distractors.append(fillers[len(distractors)])

        options = distractors[:3] + [a]
        _r.shuffle(options)
        correct_idx = options.index(a)

        out.append({"q": q, "a": a, "options": options, "correct": correct_idx})
    return out


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(
        "<div style='padding:0.2rem 0 1.4rem 0;'>"
        "<p style='font-family:Syne,sans-serif;font-weight:800;font-size:1.35rem;"
        "color:#f0ede6;margin:0;'>📄 AskMyDoc</p>"
        "<p style='font-size:0.75rem;color:#444;margin:0.25rem 0 0 0;letter-spacing:0.04em;'>"
        "AI PDF Question Answering</p></div>",
        unsafe_allow_html=True,
    )

    # 🌐 LANGUAGE SELECTOR (used for ALL audio)
    st.markdown("<span class='sidebar-label'>🌐 Audio Language</span>", unsafe_allow_html=True)

    st.selectbox(
        "",
        ["English", "Hindi", "Marathi"],
        key="audio_lang"
    )

    # 📄 Upload PDF
    st.markdown("<span class='sidebar-label'>Upload PDF</span>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["pdf"],
        label_visibility="collapsed"
    )


    # ── Handle upload ─────────────────────────────

    if uploaded is not None and st.session_state.vectorstore is None:

        with st.spinner("Indexing…"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

                tmp.write(uploaded.read())
                path = tmp.name


            docs = PyPDFLoader(path).load()

            st.session_state.full_text = "\n\n".join(
                d.page_content for d in docs
            )


            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            ).split_documents(docs)


            st.session_state.chunks = chunks


            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                load_embeddings()
            )


            # reset derived state

            for k in ["summary_bullets", "quiz_items", "chat_history"]:
                st.session_state[k] = []

            for k in ["flowchart_code", "voice_question"]:
                st.session_state[k] = ""

            st.session_state.pdf_audio = None


            os.unlink(path)


        st.success("✓ Ready!")


    elif uploaded is None and st.session_state.vectorstore is not None:

        for k, v in DEFAULTS.items():
            st.session_state[k] = v



    # ── Stats ─────────────────────────────

    if st.session_state.vectorstore:

        pages = len(
            set(c.metadata.get("page", 0) for c in st.session_state.chunks)
        )

        chunks_len = len(st.session_state.chunks)


        st.markdown(
            "<div class='stat-row'>"
            "<div class='stat-pill'><strong>" + str(pages) + "</strong>Pages</div>"
            "<div class='stat-pill'><strong>" + str(chunks_len) + "</strong>Chunks</div>"
            "</div>",
            unsafe_allow_html=True,
        )


        st.markdown("<hr>", unsafe_allow_html=True)


        # 🔊 READ PDF ALOUD

        st.markdown(
            "<span class='sidebar-label'>🔊 Read PDF Aloud</span>",
            unsafe_allow_html=True
        )


        if st.button("Generate Audio"):

            with st.spinner("Converting to speech…"):

                st.session_state.pdf_audio = text_to_audio(
                    st.session_state.full_text
                )


        if st.session_state.pdf_audio:

            st.audio(
                st.session_state.pdf_audio,
                format="audio/mp3"
            )


        st.markdown("<hr>", unsafe_allow_html=True)


        # 🎙️ VOICE QUESTION

        st.markdown(
            "<span class='sidebar-label'>🎙️ Voice Question</span>",
            unsafe_allow_html=True
        )


        st.markdown(
            "<p class='mic-hint'>Click mic → speak → stop → go to Chat tab.</p>",
            unsafe_allow_html=True,
        )


        rec = audio_recorder(
            text="",
            recording_color="#c8f04e",
            neutral_color="#333",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=3.0,
            sample_rate=16000,
        )


        if rec and rec != st.session_state.last_audio_bytes:

            st.session_state.last_audio_bytes = rec


            with st.spinner("Transcribing…"):

                t = transcribe(rec)


            if t:

                st.session_state.voice_question = t

                st.session_state.auto_answer_pending = True


                st.success("🎤 Heard: " + t + " — answering…")


                st.rerun()

            else:

                st.warning("Couldn't hear clearly. Try again.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<p class='page-title'>AskMyDoc <span class='accent'>·</span> AI PDF Q&A</p>"
    "<p class='page-sub'>Upload a PDF in the sidebar — then Chat, Summarize, Visualize or Quiz yourself.</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Welcome screen ────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
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
        "<div class='feature-item'><span>📝</span><strong>Quiz</strong>Auto-generated questions to test yourself</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

# ── Main tabs ─────────────────────────────────────────────────────────────────
else:
    tab1, tab2, tab3, tab4 = st.tabs(["💬  Chat", "📋  Summary", "🔀  Flowchart", "📝  Quiz"])

    # ══ TAB 1 — CHAT Q&A ══════════════════════════════════════════════════════
    with tab1:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>Chat Q&A with Source Highlighting</div>"
            "<p class='info-card-text'>Ask questions by typing or use the mic in the sidebar. "
            "Each answer shows which pages it came from.</p></div>",
            unsafe_allow_html=True,
        )

        # Display history
        for msg in st.session_state.chat_history:
            st.markdown(
                "<div class='chat-bubble-user'>"
                "<div class='chat-role chat-role-user'>You</div>"
                "<div class='chat-text'>" + msg["question"] + "</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='chat-bubble-bot'>"
                "<div class='chat-role chat-role-bot'>AskMyDoc</div>"
                "<div class='chat-text'>" + msg["answer"] + "</div></div>",
                unsafe_allow_html=True,
            )
            # Source chips
            if msg.get("sources"):
                chips = "".join(
                    "<span class='src-chip'>📄 Page " + str(s.metadata.get("page", 0) + 1) + "</span>"
                    for s in msg["sources"]
                )
                st.markdown("<div style='margin:0.2rem 0 0.4rem 1rem;'>" + chips + "</div>", unsafe_allow_html=True)
                with st.expander("View source chunks"):
                    for i, s in enumerate(msg["sources"]):
                        pg = s.metadata.get("page", 0) + 1
                        st.caption("Chunk " + str(i+1) + "  ·  Page " + str(pg))
                        st.markdown(
                            "<div style='background:#0f0f0f;border:1px solid #1a1a1a;border-radius:8px;"
                            "padding:0.7rem;font-size:0.82rem;color:#888;line-height:1.6;'>"
                            + s.page_content[:350].replace("<","&lt;").replace(">","&gt;")
                            + ("…" if len(s.page_content) > 350 else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

        # Audio for latest answer
        if st.session_state.chat_history:
            last_ans = st.session_state.chat_history[-1]["answer"]
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            cola, colb = st.columns([3,1])
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

        # ── Auto-answer from mic (no button needed) ───────────────────────────
        if st.session_state.auto_answer_pending and st.session_state.voice_question.strip():
            auto_q = st.session_state.voice_question.strip()
            st.session_state.auto_answer_pending = False
            st.session_state.voice_question      = ""
            with st.spinner("Finding answer…"):
                ans, srcs = answer_with_sources(auto_q, st.session_state.vectorstore)
            with st.spinner("Generating audio answer…"):
                auto_audio = text_to_audio(ans)
            st.session_state.chat_history.append({
                "question": auto_q, "answer": ans, "sources": srcs
            })
            st.session_state["last_ans_audio"] = auto_audio
            st.rerun()

        # Input row (manual typed questions still work)
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "", placeholder="Type your question… or use the 🎙️ mic in the sidebar",
                label_visibility="collapsed",
                value=st.session_state.voice_question,
                key="qinput",
            )
        with col2:
            ask = st.button("Ask →")

        if ask and question.strip():
            with st.spinner("Thinking…"):
                ans, srcs = answer_with_sources(question.strip(), st.session_state.vectorstore)
            st.session_state.chat_history.append({"question": question.strip(), "answer": ans, "sources": srcs})
            st.session_state.voice_question    = ""
            st.session_state["last_ans_audio"] = None
            st.rerun()


    # ══ TAB 2 — SUMMARY ═══════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>Smart Summary</div>"
            "<p class='info-card-text'>Condenses your PDF into concise bullet points using AI map-reduce summarization. "
            "Works best on structured documents.</p></div>",
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
        elif not st.session_state.summary_bullets and st.session_state.get("_sum_tried"):
            st.info("Couldn't extract clear points from this document. Try a more structured PDF.")


    # ══ TAB 3 — FLOWCHART ═════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>Concept Flowchart</div>"
            "<p class='info-card-text'>Extracts the 5 main topics or steps from your document and renders "
            "them as an interactive flowchart.</p></div>",
            unsafe_allow_html=True,
        )

        if st.button("🔀 Generate Flowchart"):
            with st.spinner("Extracting concepts…"):
                st.session_state.flowchart_code = generate_flowchart(st.session_state.chunks)
                st.session_state["_flow_tried"] = True

        if st.session_state.flowchart_code:
            render_mermaid(st.session_state.flowchart_code)
            with st.expander("View Mermaid diagram code"):
                st.code(st.session_state.flowchart_code, language="text")
        elif not st.session_state.flowchart_code and st.session_state.get("_flow_tried"):
            st.warning("Could not build a flowchart for this document. Try a document with clearer structure.")


    # ══ TAB 4 — QUIZ ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            "<div class='info-card'><div class='info-card-label'>MCQ Quiz</div>"
            "<p class='info-card-text'>Auto-generated multiple choice questions from your document. "
            "Pick an option for each question, then Submit to see your score.</p></div>",
            unsafe_allow_html=True,
        )

        # init state
        for _k, _v in [("quiz_answers", {}), ("quiz_result", None), ("_quiz_tried", False)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        col_gen, col_reset = st.columns([2, 1])
        with col_gen:
            if st.button("📝 Generate Quiz"):
                with st.spinner("Generating MCQ questions… (~30–45s)"):
                    st.session_state.quiz_items     = generate_quiz(st.session_state.chunks)
                    st.session_state["_quiz_tried"] = True
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
            # ── show radio-button questions ───────────────────────────────────
            disabled = bool(st.session_state["quiz_result"])  # lock after submit

            for i, item in enumerate(st.session_state.quiz_items):
                st.markdown(
                    "<div class='quiz-card'>"
                    "<div class='quiz-num'>Question " + str(i+1) + " of "
                    + str(len(st.session_state.quiz_items)) + "</div>"
                    "<div class='quiz-q'>❓ " + item["q"] + "</div></div>",
                    unsafe_allow_html=True,
                )
                opts = ["— select an option —"] + item["options"]
                chosen = st.radio(
                    "Options",
                    opts,
                    key="mcq_" + str(i),
                    label_visibility="collapsed",
                    disabled=disabled,
                )
                st.session_state["quiz_answers"][str(i)] = chosen
                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Submit button ─────────────────────────────────────────────────
            if not disabled:
                if st.button("✅ Submit & See Results", type="primary"):
                    score, results = 0, []
                    for i, item in enumerate(st.session_state.quiz_items):
                        chosen = st.session_state["quiz_answers"].get(str(i), "")
                        ok     = (chosen == item["options"][item["correct"]])
                        if ok:
                            score += 1
                        results.append({
                            "q":       item["q"],
                            "chosen":  chosen,
                            "correct": item["options"][item["correct"]],
                            "ok":      ok,
                        })
                    st.session_state["quiz_result"] = {
                        "score": score, "total": len(st.session_state.quiz_items), "details": results
                    }
                    st.rerun()

            # ── Results ───────────────────────────────────────────────────────
            if st.session_state["quiz_result"]:
                r     = st.session_state["quiz_result"]
                pct   = int((r["score"] / r["total"]) * 100)
                emoji = "🏆" if pct >= 80 else ("👍" if pct >= 50 else "📖")
                color = "#c8f04e" if pct >= 80 else ("#f0a050" if pct >= 50 else "#e05050")
                msg   = "Excellent!" if pct >= 80 else ("Good effort!" if pct >= 50 else "Keep reading!")

                # score card
                st.markdown(
                    "<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:14px;"
                    "padding:1.8rem;text-align:center;margin:1rem 0;'>"
                    "<div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;color:"
                    + color + ";line-height:1;'>" + str(pct) + "%</div>"
                    "<div style='font-size:1rem;color:" + color + ";font-family:Syne,sans-serif;"
                    "font-weight:700;margin-top:0.4rem;'>" + emoji + "  " + msg + "</div>"
                    "<div style='font-size:0.82rem;color:#555;margin-top:0.3rem;'>"
                    + str(r["score"]) + " correct out of " + str(r["total"]) + " questions</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # per-question review
                st.markdown(
                    "<p style='font-family:Syne,sans-serif;font-size:0.68rem;font-weight:700;"
                    "letter-spacing:0.12em;color:#555;text-transform:uppercase;"
                    "margin:0.5rem 0;'>Answer Review</p>",
                    unsafe_allow_html=True,
                )
                for d in r["details"]:
                    border = "#1e2d04" if d["ok"] else "#2d0e0e"
                    icon   = "✅" if d["ok"] else "❌"
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
                        + "</div>",
                        unsafe_allow_html=True,
                    )

        elif not st.session_state.quiz_items and st.session_state.get("_quiz_tried"):
            st.info("Quiz generation failed. Try a longer or more structured document.")
