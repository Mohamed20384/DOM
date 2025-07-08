import os
import io
from dotenv import load_dotenv
import streamlit as st
import PyPDF2  # Added for PDF support
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Tuple

# Streamlit page configuration MUST be first
st.set_page_config(
    page_title="مساعد المطاعم المصري",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

# Configuration
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300
    EMBEDDING_BATCH_SIZE = 10
    MAX_PREVIEW_CHARS = 5000
    PDF_FOLDER = "Restaurants_PDF"  
    SYSTEM_PROMPT = """أنت (DOM) مساعد ذكي يتحدث العربية المصرية بطلاقة.
                    و انت مواطن من مدينة دمياط الجديده لديك خبره كبيره في المطاعم في هذه المدينه.
                    ستجيب على الأسئلة بناءً على المعلومات الموجودة في مستندات PDF الخاصة بالمطاعم فقط.

                    - استخدم لغة سهلة وبسيطة كما يتحدث المصريون.
                    - إذا لم تجد الإجابة في المستندات قل "معنديش المعلومات دي للأسف".
                    - ركز على المعلومات العملية مثل العناوين، الأسعار، المأكولات، والخصائص المميزة.
                    - ممنوع تمامًا تذكر أو تحاول تخمّن عدد المطاعم أو أسمائها الكاملة.
                    - إذا سألك المستخدم عن عدد المطاعم أو أسمائها قل له: "معنديش المعلومات دي للأسف".
                    """
    
    UI_THEME = {
        "primary_color": "#FF4B4B",
        "secondary_color": "#FF9E9E",
        "background_color": "#0E1117",
        "text_color": "#FAFAFA",
        "success_color": "#00D100"
    }

# Custom CSS (same as before)
st.markdown(f"""
<style>
:root {{
    --primary: {Config.UI_THEME['primary_color']};
    --secondary: {Config.UI_THEME['secondary_color']};
    --bg: {Config.UI_THEME['background_color']};
    --text: {Config.UI_THEME['text_color']};
    --success: {Config.UI_THEME['success_color']};
}}

.reportview-container .main .block-container {{
    direction: rtl;
    text-align: right;
    max-width: 1200px;
}}

.stChatMessage {{
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
}}

.stChatMessage.user {{
    background-color: var(--bg);
    border: 1px solid var(--primary);
}}

.stChatMessage.assistant {{
    background-color: #1E1E1E;
    border: 1px solid var(--secondary);
}}

.stTextInput > div > div > input {{
    text-align: right;
    padding: 12px;
    border-radius: 15px;
}}

.stButton button {{
    background-color: var(--primary);
    color: white;
    border-radius: 15px;
    padding: 10px 24px;
}}

.stButton button:hover {{
    background-color: var(--secondary);
    color: white;
}}

.sidebar .sidebar-content {{
    background-color: var(--bg);
}}

.st-expander {{
    background-color: var(--bg);
    border: 1px solid var(--secondary);
    border-radius: 10px;
}}

.stAlert {{
    border-radius: 10px;
}}

.file-preview {{
    background-color: #1E1E1E;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}}

.file-title {{
    color: var(--primary);
    font-weight: bold;
    margin-bottom: 0.5rem;
}}

.spinner-text {{
    color: var(--secondary);
    font-size: 1.2rem;
}}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# App Header
st.title("🍽️ مساعد مطاعم دمياط (DOM)")
st.markdown("""
<div style="text-align: right; direction: rtl;">
اسألني أي سؤال عن المطاعم في دمياط الجديدة وسأجيبك بناءً على المعلومات الموجودة عندي
</div>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "أهلاً وسهلاً! أنا مساعدك الذكي للمطاعم في دمياط الجديدة. ممكن أساعدك بإيه النهاردة؟"
    })

# Cached resources
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.7
    )

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None returns
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

@st.cache_resource
def get_vectorstore():
    """Create and return the FAISS vectorstore from PDF files.
    Each restaurant (i.e., each PDF) becomes one chunk."""
    documents = []
    pdf_folder = Config.PDF_FOLDER

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        return None

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(pdf_folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    text = extract_text_from_pdf(f)
                    if text.strip():
                        documents.append((filename, text))
                    else:
                        st.warning(f"📄 ملف {filename} مفيهوش نص مقروء.")
            except Exception as e:
                st.error(f"❌ حصل خطأ مع الملف {filename}: {str(e)}")

    if not documents:
        st.error("⚠️ مفيش ملفات PDF صالحة للقراءة")
        return None

    # Make each restaurant (file) its own chunk
    texts = [text for _, text in documents]
    metadatas = [{"source": filename} for filename, _ in documents]

    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore, documents

def format_docs(docs):
    """Format retrieved documents for display"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content
        formatted.append(f"📄 المصدر: {source}\n📝 المحتوى:\n{content}\n{'='*50}")
    return "\n\n".join(formatted)

def get_retriever(vectorstore):
    """Create a retriever with similarity search"""
    return vectorstore.as_retriever(search_kwargs={"k": len(documents)})

def create_rag_chain(retriever):
    """Create the RAG chain for question answering"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", Config.SYSTEM_PROMPT),
        ("human", """السؤال: {question}
        
        المعلومات ذات الصلة:
        {context}
        
        جاوب بالعربية المصرية:""")
    ])
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

# Load and process PDF files
with st.spinner("جاري تحميل معلومات المطاعم من ملفات PDF... ⏳"):
    result = get_vectorstore()

if not result:
    st.error("⚠️ مفيش ملفات PDF موجودة في مجلد المطاعم أو لا يمكن قراءتها. رجاء ضع الملفات في مجلد 'Restaurants_PDF' وتأكد أنها تحتوي على نصوص قابلة للقراءة")
    st.stop()

vectorstore, documents = result

st.sidebar.success(f"📊 عدد المطاعم اللي تم استخراج معلوماتهم فعليًا: {vectorstore.index.ntotal}")

retriever = get_retriever(vectorstore)
rag_chain = create_rag_chain(retriever)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("اسأل سؤال عن المطاعم..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("ثواني ..."):
            try:
                response = rag_chain.invoke(question)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = "⚠️ حصل خطأ في الإجابة، حاول تاني بعد شوية"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Document information sidebar
with st.sidebar:
    st.header("📂 معلومات المطاعم المتاحة")
    st.markdown(f"""
    <div style="text-align: right; direction: rtl;">
    <p style="color: {Config.UI_THEME['success_color']};">عدد الملفات: {len(documents)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    for filename, text in documents:
        with st.expander(f"📄 {filename}", expanded=False):
            st.markdown(f"""
            <div class="file-preview">
                <div class="file-title">المحتوى المستخرج:</div>
                <div>{text[:Config.MAX_PREVIEW_CHARS] + ("..." if len(text) > Config.MAX_PREVIEW_CHARS else "")}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #888;">
    <hr>
    مساعد المطاعم المصري - إصدار 1.0
    </div>
    """, unsafe_allow_html=True)