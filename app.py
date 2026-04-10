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
import pypdf  # للتأكد

# ========== إعدادات الصفحة ==========
st.set_page_config(page_title="EliteMind - Knowledge Engine", page_icon="🏛️", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0F0F0F;
        color: #E0E0E0;
    }
    h1, h2, h3 { color: #D4AF37 !important; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(212, 175, 55, 0.2);
        margin-top: 20px;
    }
    .stButton>button {
        border: 1px solid #D4AF37;
        background-color: transparent;
        color: #D4AF37;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ EliteMind")
st.caption("المحرك الذكي النخبوي - حلل المستندات، المواقع، ويوتيوب")

# ========== الشريط الجانبي ==========
with st.sidebar:
    st.header("⚙️ الإعدادات")
    groq_key = st.text_input("🔑 مفتاح Groq API", type="password")
    source_type = st.selectbox("نوع المصدر", ["ملف (PDF/DOCX/TXT)", "رابط موقع ويب", "فيديو يوتيوب"])
    st.info("🚀 يستخدم نموذج llama-3.3-70b-versatile")

# ========== إدخال المصدر ==========
source_input = None
if source_type == "ملف (PDF/DOCX/TXT)":
    source_input = st.file_uploader("ارفع ملفك", type=["pdf", "docx", "txt"])
elif source_type == "رابط موقع ويب":
    source_input = st.text_input("أدخل رابط الموقع")
elif source_type == "فيديو يوتيوب":
    source_input = st.text_input("أدخل رابط يوتيوب")

question = st.text_area("💬 اسأل سؤالك هنا:", height=100, placeholder="مثال: حلل هذا الملف وأعطني أهم النقاط...")

# متغير لتخزين الإجابة
answer_text = ""

if st.button("✨ حلل وأجب", type="primary"):
    if not groq_key:
        st.error("❌ الرجاء إدخال مفتاح Groq API")
    elif not source_input or not question:
        st.error("❌ الرجاء إدخال المصدر والسؤال")
    else:
        with st.spinner("💎 جاري تحليل المصدر..."):
            try:
                docs = []
                # ------------------- تحميل الملفات -------------------
                if source_type == "ملف (PDF/DOCX/TXT)":
                    # حفظ الملف المؤقت
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{source_input.name.split('.')[-1]}") as tmp:
                        tmp.write(source_input.getvalue())
                        tmp_path = tmp.name
                    
                    ext = source_input.name.split(".")[-1].lower()
                    if ext == "pdf":
                        # محاولة تحميل PDF مع التحقق من وجود نص
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        # إذا كان المستند فارغاً (لا نص)، جرب طريقة أخرى
                        if not docs or all(len(doc.page_content.strip()) < 50 for doc in docs):
                            st.warning("⚠️ يبدو أن هذا الملف ممسوح ضوئياً (scanned) أو لا يحتوي على نص قابل للقراءة. الرجاء استخدام ملف نصي أو PDF قابل للبحث.")
                            os.unlink(tmp_path)
                            st.stop()
                    elif ext == "docx":
                        loader = UnstructuredWordDocumentLoader(tmp_path)
                        docs = loader.load()
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
                    st.error("لم يتم العثور على أي محتوى في المصدر المحدد.")
                    st.stop()
                
                # عرض عينة من النص المستخرج للتحقق
                with st.expander("📄 معاينة النص المستخرج (أول 500 حرف)"):
                    st.write(docs[0].page_content[:500] + "...")
                
                # ------------------- تقطيع النص -------------------
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)
                
                if not splits:
                    st.error("فشل في تقسيم النص إلى أجزاء. الرجاء التحقق من صحة الملف.")
                    st.stop()
                
                # ------------------- Embeddings و Vectorstore -------------------
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                
                # ------------------- إعداد Groq (النموذج الجديد) -------------------
                llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile", groq_api_key=groq_key)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})  # زيادة عدد الشظايا
                )
                
                answer_text = qa_chain.run(question)
                
                # ------------------- عرض الإجابة -------------------
                st.markdown(f'<div class="glass-card"><h3>📜 الإجابة المستخلصة:</h3><p>{answer_text}</p></div>', unsafe_allow_html=True)
                
                # ------------------- زر تحميل الإجابة (بديل الطباعة) -------------------
                if answer_text:
                    # إنشاء محتوى للتحميل
                    report_content = f"""# تقرير EliteMind
                    
**السؤال:** {question}

**المصدر:** {source_input if isinstance(source_input, str) else source_input.name}

**التاريخ:** {st.session_state.get('timestamp', 'اليوم')}

**الإجابة:**
{answer_text}

---
*تم إنشاؤه بواسطة EliteMind - محرك المعرفة النخبوي*
"""
                    st.download_button(
                        label="📥 تحميل الإجابة كملف نصي (.txt)",
                        data=report_content,
                        file_name="EliteMind_Report.txt",
                        mime="text/plain",
                        key="download_btn"
                    )
            
            except Exception as e:
                st.error(f"⚠️ حدث خطأ: {str(e)}")
                st.info("تأكد من أن ملف PDF يحتوي على نصوص قابلة للاستخراج. إذا كان ممسوحاً، استخدم ملفاً نصياً بدلاً من ذلك.")
