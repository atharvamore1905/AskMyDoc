import streamlit as st
import tempfile
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AskMyDoc", page_icon="📄", layout="centered")

# ── CSS (assigned to a variable first to avoid Python 3.14 parsing quirks) ───
CSS = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');"
    "html,body,[class*='css']{font-family:'Inter',sans-serif;}"
    ".stApp{background:#0d0d0d;color:#f0ede6;}"
    ".main{background-color:#0d0d0d;}"
    ".title-wrap{text-align:center;padding:2.5rem 0 1.5rem 0;}"
    ".title-wrap h1{font-family:'Syne',sans-serif;font-weight:800;font-size:2.6rem;"
    "color:#f0ede6;letter-spacing:-1px;margin-bottom:0.3rem;}"
    ".title-wrap p{font-size:0.95rem;color:#777;font-weight:300;}"
    ".accent{color:#c8f04e;}"
    "[data-testid='stFileUploader']{background:#1a1a1a;border:1.5px dashed #333;"
    "border-radius:12px;padding:1rem;}"
    "[data-testid='stTextInput'] input{background:#1a1a1a !important;"
    "border:1.5px solid #2a2a2a !important;border-radius:10px !important;"
    "color:#f0ede6 !important;font-size:0.95rem !important;padding:0.75rem 1rem !important;}"
    "[data-testid='stTextInput'] input:focus{border-color:#c8f04e !important;"
    "box-shadow:0 0 0 2px rgba(200,240,78,0.12) !important;}"
    ".stButton>button{background:#c8f04e !important;color:#0d0d0d !important;"
    "font-family:'Syne',sans-serif !important;font-weight:700 !important;"
    "border:none !important;border-radius:10px !important;"
    "padding:0.65rem 2rem !important;width:100% !important;}"
    ".answer-card{background:#141414;border:1px solid #2a2a2a;"
    "border-left:3px solid #c8f04e;border-radius:12px;padding:1.4rem 1.6rem;margin-top:1.2rem;}"
    ".answer-label{font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:700;"
    "letter-spacing:0.12em;color:#c8f04e;text-transform:uppercase;margin-bottom:0.6rem;}"
    ".answer-text{font-size:1.05rem;color:#f0ede6;line-height:1.65;}"
    "hr{border-color:#1e1e1e !important;margin:1.5rem 0 !important;}"
    ".section-label{font-family:'Syne',sans-serif;font-size:0.72rem;font-weight:600;"
    "letter-spacing:0.1em;color:#555;text-transform:uppercase;margin-bottom:0.5rem;}"
    "</style>"
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
TITLE_HTML = (
    "<div class='title-wrap'>"
    "<h1>AskMyDoc <span class='accent'>·</span> AI PDF Question Answering</h1>"
    "<p>Upload a document &mdash; ask anything about it.</p>"
    "</div>"
)
st.markdown(TITLE_HTML, unsafe_allow_html=True)
st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    name      = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tokenizer, model

# ── QA logic ──────────────────────────────────────────────────────────────────
def answer_question(question, vectorstore):
    docs    = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)
    prompt  = (
        "Answer the question concisely and accurately using only the context below.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n"
        "Answer:"
    )
    tokenizer, model = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.markdown("<p class='section-label'>&#9312; Upload your PDF</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("Reading and indexing your document…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        docs     = PyPDFLoader(tmp_path).load()
        chunks   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        embeddings = load_embeddings()
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        os.unlink(tmp_path)

    st.success("✓ Document indexed — " + str(len(chunks)) + " chunks ready.")

elif uploaded_file is None and st.session_state.vectorstore is not None:
    st.session_state.vectorstore = None

# ── Step 2: Question ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p class='section-label'>&#9313; Ask a question</p>", unsafe_allow_html=True)
question    = st.text_input("", placeholder="e.g. What is the main topic of this document?", label_visibility="collapsed")
ask_clicked = st.button("Get Answer")

# ── Step 3: Answer ────────────────────────────────────────────────────────────
if ask_clicked:
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking…"):
            answer = answer_question(question, st.session_state.vectorstore)
        if answer:
            answer_html = (
                "<div class='answer-card'>"
                "<div class='answer-label'>Answer</div>"
                "<div class='answer-text'>" + answer + "</div>"
                "</div>"
            )
            st.markdown(answer_html, unsafe_allow_html=True)
        else:
            st.info("No answer could be generated. Try rephrasing your question.")