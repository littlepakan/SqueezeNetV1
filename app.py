import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pickle
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Pes Planus AI Diagnosis System",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for medical dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box-normal {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .result-box-flatfoot {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .status-text {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.markdown('<div class="main-header">🦶 ระบบวินิจฉัยภาวะเท้าแบนด้วย AI (SqueezeNet + SVM)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Diagnosis of Pes Planus via Calcaneal Inclusion Angle Feature Extraction</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.header("⚙️ การตั้งค่าระบบ & โหลดโมเดล")

# Model Loading Mode
model_source = st.sidebar.radio(
    "แหล่งที่มาของไฟล์โมเดล (.pkl):",
    ("ค้นหาจากโฟลเดอร์ปัจจุบัน", "อัปโหลดไฟล์โมเดลด้วยตนเอง")
)

svm_model = None
rfe_selector = None

if model_source == "ค้นหาจากโฟลเดอร์ปัจจุบัน":
    # Search for available fold models
    available_folds = []
    for f in range(1, 6):
        svm_path = f"Fold_{f}_best_svm.pkl"
        rfe_path = f"Fold_{f}_rfe_selector.pkl"
        if os.path.exists(svm_path) and os.path.exists(rfe_path):
            available_folds.append(f"Fold_{f}")
    
    if available_folds:
        selected_fold = st.sidebar.selectbox("เลือก Fold ของโมเดล:", available_folds)
        svm_file = f"{selected_fold}_best_svm.pkl"
        rfe_file = f"{selected_fold}_rfe_selector.pkl"
        
        try:
            with open(svm_file, 'rb') as f:
                svm_model = pickle.load(f)
            with open(rfe_file, 'rb') as f:
                rfe_selector = pickle.load(f)
            st.sidebar.success(f"โหลด {selected_fold} สำเร็จ!")
        except Exception as e:
            st.sidebar.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    else:
        st.sidebar.warning("⚠️ ไม่พบไฟล์ .pkl ในโฟลเดอร์ โปรดวางไฟล์ Fold_X_best_svm.pkl ในไดเรกทอรีเดียวกับ app.py")

else:
    # Manual Upload
    uploaded_svm = st.sidebar.file_uploader("อัปโหลดไฟล์ SVM Model (.pkl)", type=["pkl"], key="svm")
    uploaded_rfe = st.sidebar.file_uploader("อัปโหลดไฟล์ RFE Selector (.pkl)", type=["pkl"], key="rfe")
    
    if uploaded_svm and uploaded_rfe:
        try:
            svm_model = pickle.load(uploaded_svm)
            rfe_selector = pickle.load(uploaded_rfe)
            st.sidebar.success("โหลดไฟล์โมเดลอัปโหลดสำเร็จ!")
        except Exception as e:
            st.sidebar.error(f"ไฟล์โมเดลไม่ถูกต้อง: {e}")

# ---------------------------------------------------------
# Load SqueezeNet Model (Cached for Speed)
# ---------------------------------------------------------
@st.cache_resource
def load_squeezenet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    model = model.to(device)
    model.eval()
    return model, device

squeezenet_model, device = load_squeezenet()

# Image Preprocessing Functions
def apply_median_filter(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    median_img = cv2.medianBlur(img_cv, 3)
    final_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_img)

eval_transforms = transforms.Compose([
    transforms.Lambda(apply_median_filter),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# Main Interface: File Upload & Prediction
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 อัปโหลดภาพเอกซเรย์เท้า (X-ray)")
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพเอกซเรย์มุมด้านข้าง (Lateral View)...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="ภาพเอกซเรย์ต้นฉบับ", use_container_width=True)
        
        # Display Median Filtered preview
        filtered_image = apply_median_filter(image)
        with st.expander("🔍 ดูภาพหลังทำ Median Filtering (ลด Noise)"):
            st.image(filtered_image, caption="ภาพหลังผ่าน Median Filter (3x3)", use_container_width=True)

with col2:
    st.subheader("📊 ผลการวิเคราะห์และวินิจฉัย")
    
    if uploaded_file is None:
        st.info("👈 กรุณาอัปโหลดภาพเอกซเรย์ที่เมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์")
    elif svm_model is None or rfe_selector is None:
        st.error("⚠️ กรุณาโหลดไฟล์โมเดล (.pkl) ในเมนู Sidebar ก่อนทำการวิเคราะห์")
    else:
        if st.button("🚀 เริ่มประมวลผลและวินิจฉัย", type="primary", use_container_width=True):
            with st.spinner("⏳ กำลังสกัด Feature ด้วย SqueezeNet และประมวลผลผ่าน SVM..."):
                # 1. Transform Image
                img_tensor = eval_transforms(image).unsqueeze(0).to(device)
                
                # 2. Extract Deep Features via SqueezeNet (512 dimensions)
                with torch.no_grad():
                    features = squeezenet_model(img_tensor).cpu().numpy()
                
                # 3. Apply RFE Selection (512 -> 200 features)
                features_opt = rfe_selector.transform(features)
                
                # 4. Predict via SVM
                prediction = svm_model.predict(features_opt)[0]
                
                # Get prediction probabilities or decision function if available
                prob = None
                if hasattr(svm_model, "predict_proba"):
                    try:
                        prob = svm_model.predict_proba(features_opt)[0]
                    except Exception:
                        pass
                
                # Display Results
                st.markdown("---")
                if prediction == 1: # Pes Planus
                    st.markdown("""
                    <div class="result-box-flatfoot">
                        <div class="status-text" style="color: #DC2626;">🚨 ผลวินิจฉัย: มีภาวะเท้าแบน (Pes Planus)</div>
                        <p style="color: #7F1D1D; margin-top: 0.5rem;">
                            แบบจำลองตรวจพบโครงสร้างมุมกระดูกส้นเท้า (Calcaneal Angle) ที่มีความลาดเอียงเข้าข่ายภาวะเท้าแบน
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else: # Normal
                    st.markdown("""
                    <div class="result-box-normal">
                        <div class="status-text" style="color: #059669;">✅ ผลวินิจฉัย: โครงสร้างเท้าปกติ (Normal)</div>
                        <p style="color: #065F46; margin-top: 0.5rem;">
                            แบบจำลองไม่พบความผิดปกติของมุมส่วนโค้งกระดูกเท้า ภาพเอกซเรย์อยู่ในเกณฑ์ปกติ
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability display if supported
                if prob is not None:
                    st.write("### 🎯 ค่าความน่าจะเป็น (Confidence Score)")
                    p_normal = prob[0] * 100
                    p_flatfoot = prob[1] * 100
                    
                    st.write(f"**ปกติ (Normal):** {p_normal:.2f}%")
                    st.progress(int(p_normal))
                    
                    st.write(f"**เท้าแบน (Pes Planus):** {p_flatfoot:.2f}%")
                    st.progress(int(p_flatfoot))
                
                # Metric Information
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Feature Extractor", "SqueezeNet v1.1")
                m2.metric("Feature Selection", "RFE (200 Features)")
                m3.metric("Classifier", f"SVM ({getattr(svm_model, 'kernel', 'Linear/RBF').upper()})")

st.markdown("---")
st.caption("👨‍⚕️ *หมายเหตุ: ระบบ AI นี้จัดทำขึ้นเพื่อช่วยสนับสนุนการวินิจฉัยเบื้องต้น การวินิจฉัยขั้นสุทธิควรได้รับการยืนยันโดยแพทย์ผู้เชี่ยวชาญด้านออร์โธปิดิกส์/รังสีแพทย์*")