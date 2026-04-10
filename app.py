import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    YoutubeLoader, UnstructuredURLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# ========== إعدادات الصفحة والفخامة ==========
st.set_page_config(page_title="EliteMind - Knowledge Engine", page_icon="🏛️", layout="wide")

# إخفاء شريط Streamlit الافتراضي
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# التصميم الذهبي المطفي والزجاجي + تخصيص الطباعة لتصدير PDF
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0F0F0F;
        color: #E0E0E0;
    }
    .stApp { background-color: #0F0F0F; }
    
    h1, h2, h3 { color: #D4AF37 !important; font-weight: 700; }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(212, 175, 55, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-top: 20px;
    }

    .stButton>button {
        border: 1px solid #D4AF37;
        background-color: transparent;
        color: #D4AF37;
        border-radius: 10px;
        padding: 10px 25px;
        transition: all 0.4s ease;
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: #000;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.4);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        color: white;
    }
    .stSelectbox>div>div {
        background-color: #1E1E1E;
        border-radius: 10px;
    }
    
    /* تنسيق الطباعة لتصدير PDF */
    @media print {
        .no-print, .stButton, .stTextInput, .stTextArea, .stSelectbox, .stFileUploader, .stRadio, .sidebar .sidebar-content, header, footer, #MainMenu {
            display: none !important;
        }
        .glass-card, .main .block-container {
            background: white !important;
            color: black !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        h1, h2, h3 {
            color: black !important;
        }
        body {
            background: white;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ========== الواجهة الرئيسية ==========
st.title("🏛️ EliteMind")
st.caption("المحرك الذكي النخبوي: حلل المستندات، المواقع، وفيديوهات يوتيوب بذكاء خارق وسرعة فائقة.")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    groq_key = st.text_input("🔑 مفتاح Groq API", type="password", help="احصل عليه من console.groq.com")
    source_type = st.selectbox("نوع المصدر", ["ملف (PDF/DOCX/TXT)", "رابط موقع ويب", "فيديو يوتيوب"])
    st.markdown("---")
    st.info("🚀 يستخدم نموذج Llama-3-70B عبر Groq لسرعة استثنائية وجودة نخبوية.")

# إدخال المصدر
source_input = None
if source_type == "ملف (PDF/DOCX/TXT)":
    source_input = st.file_uploader("ارفع ملفك", type=["pdf", "docx", "txt"])
elif source_type == "رابط موقع ويب":
    source_input = st.text_input("أدخل رابط الموقع (URL)")
elif source_type == "فيديو يوتيوب":
    source_input = st.text_input("أدخل رابط فيديو يوتيوب")

question = st.text_area("💬 اسأل سؤالك هنا:", height=100)

# متغير لحفظ الإجابة
answer_text = ""

col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button("✨ حلل وأجب", type="primary")
with col2:
    pass

if analyze_btn:
    if not groq_key:
        st.error("❌ الرجاء إدخال مفتاح Groq API أولاً.")
    elif not source_input or not question:
        st.error("❌ الرجاء إدخال المصدر والسؤال.")
    else:
        with st.spinner("💎 جاري تفعيل العقل الاصطناعي ومعالجة البيانات..."):
            try:
                docs = []
                # معالجة الملفات
                if source_type == "ملف (PDF/DOCX/TXT)":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{source_input.name.split('.')[-1]}") as tmp:
                        tmp.write(source_input.getvalue())
                        tmp_path = tmp.name
                    ext = source_input.name.split(".")[-1].lower()
                    if ext == "pdf":
                        loader = PyPDFLoader(tmp_path)
                    elif ext == "docx":
                        loader = UnstructuredWordDocumentLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding='utf-8')
                    docs = loader.load()
                    os.unlink(tmp_path)
                
                elif source_type == "رابط موقع ويب":
                    loader = UnstructuredURLLoader(urls=[source_input])
                    docs = loader.load()
                
                elif source_type == "فيديو يوتيوب":
                    loader = YoutubeLoader.from_youtube_url(source_input, add_video_info=True)
                    docs = loader.load()
                
                if not docs:
                    st.error("لم يتم العثور على محتوى في المصدر المحدد.")
                    st.stop()
                
                # تقطيع النصوص
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)
                
                # Embeddings مجانية
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                
                # إعداد Groq
                llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
                
                answer_text = qa_chain.run(question)
                
                st.markdown(f'<div id="answer-section" class="glass-card"><h3>📜 الإجابة المستخلصة:</h3><p>{answer_text}</p></div>', unsafe_allow_html=True)
                
                # زر تصدير PDF
                pdf_btn = st.button("📄 تصدير الإجابة كـ PDF (طباعة)", key="pdf_btn")
                if pdf_btn:
                    # استخدام JavaScript لفتح نافذة الطباعة
                    st.markdown("""
                        <script>
                        window.print();
                        </script>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"⚠️ حدث خطأ: {str(e)}")
