import os
import io
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Streamlit page configuration MUST be first
st.set_page_config(
    page_title="مساعد المطاعم المصري",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

def get_restaurant_names_from_folder(folder_path: str) -> List[str]:
    names = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            name = os.path.splitext(file)[0].strip()
            names.append(name)
    return names

PDF_FOLDER_PATH = "Restaurants_PDF"
restaurant_names = get_restaurant_names_from_folder(PDF_FOLDER_PATH)
restaurant_list_text = "\n".join([f"- {name}" for name in restaurant_names])

# Configuration
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300
    EMBEDDING_BATCH_SIZE = 10
    MAX_PREVIEW_CHARS = 5000
    PDF_FOLDER = PDF_FOLDER_PATH
    SYSTEM_PROMPT = f"""
                        أنت مساعد ذكي اسمك DOM، شغال في تطبيق مطاعم دمياط. بتتكلم باللهجة المصرية، ومن أهل دمياط الجديدة وعندك خبرة كبيرة بكل المطاعم اللي فيها.

                        دورك إنك تجاوب على أسئلة المستخدمين بناءً على المعلومات الموجودة في ملفات PDF الخاصة بالمطاعم فقط.

                        تعليمات مهمة:
                        - لما المستخدم يسألك عن نفسك أو يقولك "إنت مين؟" أو "عرفني بنفسك"، وقتها بس عرف نفسك وقل: "أنا DOM، مساعدك الذكي في مطاعم دمياط".
                        - لو المستخدم سألك عن المطاعم أو أي حاجة تانية، متعرفش عن نفسك خالص وادخل في الموضوع على طول.
                        - ردودك لازم تكون باللهجة المصرية، وبأسلوب بسيط وواضح.
                        - لو الإجابة مش موجودة في الملفات، قول: "معنديش المعلومات دي للأسف".
                        - ركز على المعلومات المفيدة زي العناوين، الأسعار، الأكلات، والمميزات الخاصة بكل مطعم.
                        - لو حد سألك عن عدد المطاعم أو أسمائها، جاوبه كده:
                        "المطاعم اللي عندي حالياً هي:
                        {restaurant_list_text}"
                    """
    UI_THEME = {
        "primary_color": "#FF4B4B",
        "secondary_color": "#FF9E9E",
        "background_color": "#0E1117",
        "text_color": "#FAFAFA",
        "success_color": "#00D100"
    }

# Custom CSS
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "أهلاً وسهلاً! أنا مساعدك الذكي للمطاعم في دمياط الجديدة. ممكن أساعدك بإيه النهاردة؟"
    })

if "token_usage" not in st.session_state:
    st.session_state.token_usage = []

# Token counting function
def count_tokens(text: str) -> int:
    """Simple token estimation (4 chars ≈ 1 token)"""
    return max(1, len(text) // 4)

# Cached resources
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.7,
        max_output_tokens=3000
    )

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )

def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    lines = [line for line in text.split('\n') if len(line.strip()) > 1]
    cleaned = '\n'.join(line for line in lines if not line.strip().isdigit())
    return cleaned.replace('-\n', '')

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

@st.cache_resource
def get_vectorstore():
    """Create and return the FAISS vectorstore from PDF files"""
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

def get_compressed_retriever(base_retriever):
    embeddings = get_embeddings()
    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

# def get_retriever(vectorstore):
#     bm25_retriever = BM25Retriever.from_texts(
#         [doc[1] for doc in documents],
#         metadatas=[{"source": doc[0]} for doc in documents]
#     )
#     bm25_retriever.k = 5
    
#     faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
#     return EnsembleRetriever(
#         retrievers=[bm25_retriever, faiss_retriever],
#         weights=[0.4, 0.6]
#     )

def get_retriever(vectorstore):
    bm25_retriever = BM25Retriever.from_texts(
        [doc[1] for doc in documents],
        metadatas=[{"source": doc[0]} for doc in documents]
    )
    bm25_retriever.k = 5

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    return get_compressed_retriever(ensemble)

@st.cache_resource
def get_conversation_chain(_retriever):
    """Create conversation chain with memory"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    llm = get_llm()
    
    # Create a prompt template that includes the restaurant list
    # prompt_template = ChatPromptTemplate.from_messages([
    #     ("system", Config.SYSTEM_PROMPT.format(restaurant_list_text=restaurant_list_text)),
    #     ("human", """السؤال: {question}
        
    #     المعلومات ذات الصلة:
    #     {context}
        
    #     جاوب بالعربية المصرية:""")
    # ])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", Config.SYSTEM_PROMPT.format(restaurant_list_text=restaurant_list_text)),
        ("user", "السؤال السابق:\n{chat_history}\n\nالسؤال الحالي:\n{question}\n\nالمعلومات ذات الصلة:\n{context}\n\nجاوب بالعربية المصرية:")
    ])

    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt_template},
        get_chat_history=lambda h: h,
        verbose=True
    )

def load_no_eshop_restaurants(file_path: str = "no_eshop_restaurants.txt") -> List[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

no_eshop_restaurants = load_no_eshop_restaurants()

# Load and process PDF files
with st.spinner("جاري تحميل معلومات المطاعم من ملفات PDF... ⏳"):
    result = get_vectorstore()

if not result:
    st.error("⚠️ مفيش ملفات PDF موجودة في مجلد المطاعم أو لا يمكن قراءتها. رجاء ضع الملفات في مجلد 'Restaurants_PDF' وتأكد أنها تحتوي على نصوص قابلة للقراءة")
    st.stop()

vectorstore, documents = result

st.sidebar.success(f"📊 عدد المطاعم اللي تم استخراج معلوماتهم فعليًا: {vectorstore.index.ntotal}")

retriever = get_retriever(vectorstore)
conversation_chain = get_conversation_chain(retriever)  # Pass the retriever directly

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
                # === Check if question is about a restaurant with no eShop ===
                for name in no_eshop_restaurants:
                    if name in question:
                        fallback_msg = f"للأسف معندناش معلومات عن المطعم '{name}' لأنه مش مشترك في أبلكيشن مطاعم دمياط"
                        st.markdown(fallback_msg)
                        st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                        st.stop()

                # Get response with conversation memory
                result = conversation_chain({"question": question})
                response = result["answer"]
                
                # Get context for token counting
                relevant_docs = retriever.get_relevant_documents(question)
                context = format_docs(relevant_docs)

                # Prepare chat history string for token counting
                chat_history_str = "\n".join(
                    [f"{type(m).__name__}: {m.content}" for m in conversation_chain.memory.chat_memory.messages]
                )

                # Calculate and store token usage
                chat_history_tokens = count_tokens(chat_history_str)
                question_tokens = count_tokens(question)
                context_tokens = count_tokens(context)
                system_tokens = count_tokens(Config.SYSTEM_PROMPT.format(restaurant_list_text=restaurant_list_text))
                response_tokens = count_tokens(response)
                total_tokens = (
                    question_tokens + context_tokens + system_tokens + chat_history_tokens + response_tokens
                )

                st.session_state.token_usage.append({
                    "question": question,
                    "question_tokens": question_tokens,
                    "context_tokens": context_tokens,
                    "system_tokens": system_tokens,
                    "chat_history_tokens": chat_history_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "timestamp": datetime.now().isoformat()
                })

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.exception(e)
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
    
    # Token usage display
    if st.session_state.token_usage:
        latest = st.session_state.token_usage[-1]
        st.markdown(f"""
        <div style="text-align: right; direction: rtl; margin-top: 1rem; padding: 1rem; background-color: #1E1E1E; border-radius: 10px;">
            <p><b>Token Usage:</b></p>
            <p>السؤال: {latest['question_tokens']}</p>
            <p>المعلومات: {latest['context_tokens']}</p>
            <p>النظام: {latest['system_tokens']}</p>
            <p>الرد: {latest['response_tokens']}</p>
            <p>التاريخ: {latest['chat_history_tokens']}</p>
            <p><b>المجموع: {latest['total_tokens']}</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #888;">
    <hr>
    مساعد المطاعم المصري - إصدار 1.0
    </div>
    """, unsafe_allow_html=True)
