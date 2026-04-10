import streamlit as st
import os
import tempfile
import base64
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    YoutubeLoader, UnstructuredURLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from fpdf import FPDF
import re

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

# التصميم الذهبي
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
    </style>
    """, unsafe_allow_html=True)

# ========== دوال مساعدة ==========
def clean_text_for_pdf(text):
    """تنظيف النص من الرموز غير المدعومة في PDF"""
    text = re.sub(r'[^\x00-\x7F\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', ' ', text)
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf(answer_text, question_text):
    """إنشاء ملف PDF من الإجابة"""
    pdf = FPDF()
    pdf.add_page()
    # إضافة خط يدعم العربية (اختياري، لكنه يعمل مع latin-1)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    
    # عنوان
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="EliteMind - Report", ln=True, align='C')
    pdf.ln(10)
    
    # السؤال
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Question: {clean_text_for_pdf(question_text[:100])}", ln=True)
    pdf.ln(5)
    
    # الإجابة
    pdf.set_font("Arial", size=11)
    cleaned_answer = clean_text_for_pdf(answer_text)
    # تقسيم النص الطويل لأسطر
    pdf.multi_cell(0, 8, txt=cleaned_answer)
    
    # حفظ PDF في ملف مؤقت
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# ========== الواجهة الرئيسية ==========
st.title("🏛️ EliteMind")
st.caption("المحرك الذكي النخبوي: حلل المستندات، المواقع، وفيديوهات يوتيوب بذكاء خارق وسرعة فائقة.")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    groq_key = st.text_input("🔑 مفتاح Groq API", type="password", 
                             help="احصل عليه من console.groq.com",
                             value=st.secrets.get("GROQ_API_KEY", ""))
    source_type = st.selectbox("نوع المصدر", ["ملف (PDF/DOCX/TXT)", "رابط موقع ويب", "فيديو يوتيوب"])
    st.markdown("---")
    st.info("🚀 يستخدم نموذج Llama-3.3-70B-Versatile عبر Groq لسرعة استثنائية وجودة نخبوية.")

source_input = None
if source_type == "ملف (PDF/DOCX/TXT)":
    source_input = st.file_uploader("ارفع ملفك", type=["pdf", "docx", "txt"])
elif source_type == "رابط موقع ويب":
    source_input = st.text_input("أدخل رابط الموقع (URL)")
elif source_type == "فيديو يوتيوب":
    source_input = st.text_input("أدخل رابط فيديو يوتيوب")

question = st.text_area("💬 اسأل سؤالك هنا:", height=100)

if st.button("✨ حلل وأجب", type="primary"):
    if not groq_key:
        st.error("❌ الرجاء إدخال مفتاح Groq API أولاً.")
    elif not source_input or not question:
        st.error("❌ الرجاء إدخال المصدر والسؤال.")
    else:
        with st.spinner("💎 جاري تفعيل العقل الاصطناعي ومعالجة البيانات..."):
            try:
                docs = []
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
                
                # تقطيع النصوص مع حجم chunk مناسب
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # استخدام نموذج تضمين أقوى (مع دعم أفضل للغة التقنية)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                
                # النموذج الجديد من Groq
                llm = ChatGroq(temperature=0.2, model="llama-3.3-70b-versatile", groq_api_key=groq_key)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=False
                )
                
                answer_text = qa_chain.run(question)
                
                # عرض الإجابة
                st.markdown(f'<div id="answer-section" class="glass-card"><h3>📜 الإجابة المستخلصة:</h3><p>{answer_text}</p></div>', unsafe_allow_html=True)
                
                # زر تصدير PDF باستخدام FPDF (يعمل 100%)
                if st.button("📄 تصدير الإجابة كـ PDF", key="pdf_btn"):
                    pdf_path = create_pdf(answer_text, question)
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="EliteMind_Report.pdf">📥 اضغط لتحميل ملف PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    os.unlink(pdf_path)
                
            except Exception as e:
                st.error(f"⚠️ حدث خطأ: {str(e)}")
