import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pickle
import os
import pandas as pd

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Pes Planus AI Batch Diagnosis System",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for medical dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    .status-tp { color: #15803D; font-weight: bold; } /* Green */
    .status-tn { color: #0369A1; font-weight: bold; } /* Blue */
    .status-fp { color: #D97706; font-weight: bold; } /* Orange */
    .status-fn { color: #DC2626; font-weight: bold; } /* Red */
</style>
""", unsafe_allow_html=True)

# Application Title
st.markdown('<div class="main-header">🦶 ระบบวินิจฉัยภาวะเท้าแบนด้วย AI (Batch Processing & Validation)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Image Deep Learning Diagnosis with Confusion Matrix Analysis (SqueezeNet + SVM)</div>', unsafe_allow_html=True)

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

# Helper Function: Classify Outcome Matrix
def get_matrix_status(actual, pred):
    if actual is None:
        return "N/A"
    if actual == 1 and pred == 1:
        return "True Positive (TP)"
    elif actual == 0 and pred == 0:
        return "True Negative (TN)"
    elif actual == 0 and pred == 1:
        return "False Positive (FP)"
    elif actual == 1 and pred == 0:
        return "False Negative (FN)"

# ---------------------------------------------------------
# Main Interface: Batch File Upload & Processing
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 1. อัปโหลดรูปภาพ และ ไฟล์เฉลย (CSV)")
    
    # Upload Multiple Images
    uploaded_files = st.file_uploader(
        "เลือกภาพเอกซเรย์ (อัปโหลดได้หลายรูปพร้อมกัน)...", 
        type=["jpg", "jpeg", "png", "bmp"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"🖼️ เลือกไฟล์ภาพทั้งหมด: {len(uploaded_files)} ไฟล์")
    
    # Upload Ground Truth CSV File
    st.markdown("---")
    uploaded_csv = st.file_uploader("อัปโหลดไฟล์เฉลย Ground Truth (.csv) [Optional]", type=["csv"])
    
    ground_truth_dict = {}
    if uploaded_csv is not None:
        try:
            df_gt = pd.read_csv(uploaded_csv)
            st.write("📋 **ตัวอย่างข้อมูลในไฟล์เฉลย (CSV):**")
            st.dataframe(df_gt.head(3), use_container_width=True)
            
            # Select relevant columns dynamically
            col_filename = st.selectbox("เลือกคอลัมน์ที่เป็น **ชื่อไฟล์**:", df_gt.columns, index=0)
            col_label = st.selectbox("เลือกคอลัมน์ที่เป็น **ค่าเฉลย** (1 = เท้าแบน, 0 = ปกติ):", df_gt.columns, index=min(1, len(df_gt.columns)-1))
            
            # Build search mapping dict
            for _, row in df_gt.iterrows():
                fname = str(row[col_filename]).strip()
                label_val = int(row[col_label])
                ground_truth_dict[fname] = label_val
                
            st.success(f"แมปข้อมูลเฉลยสำเร็จทั้งหมด {len(ground_truth_dict)} รายการ")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")

with col2:
    st.subheader("📊 2. ประมวลผลและแสดงผลการวินิจฉัย")
    
    if not uploaded_files:
        st.info("👈 กรุณาอัปโหลดภาพเอกซเรย์ที่เมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์")
    elif svm_model is None or rfe_selector is None:
        st.error("⚠️ กรุณาโหลดไฟล์โมเดล (.pkl) ในเมนู Sidebar ก่อน")
    else:
        if st.button("🚀 เริ่มการประมวลผลทั้งหมด (Batch Prediction)", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            tp, tn, fp, fn = 0, 0, 0, 0
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"⏳ กำลังวิเคราะห์รูปที่ {idx+1}/{len(uploaded_files)}: {file.name}")
                
                # Open image
                image = Image.open(file).convert('RGB')
                
                # 1. Feature Extraction via SqueezeNet
                img_tensor = eval_transforms(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = squeezenet_model(img_tensor).cpu().numpy()
                
                # 2. RFE Selection
                features_opt = rfe_selector.transform(features)
                
                # 3. SVM Prediction
                pred_label = int(svm_model.predict(features_opt)[0])
                
                # Probability / Confidence calculation
                confidence = None
                if hasattr(svm_model, "predict_proba"):
                    try:
                        prob = svm_model.predict_proba(features_opt)[0]
                        confidence = prob[pred_label] * 100
                    except Exception:
                        pass
                
                # Match Ground Truth
                actual_label = ground_truth_dict.get(file.name, None)
                matrix_status = get_matrix_status(actual_label, pred_label)
                
                # Accumulate confusion matrix metrics
                if matrix_status == "True Positive (TP)": tp += 1
                elif matrix_status == "True Negative (TN)": tn += 1
                elif matrix_status == "False Positive (FP)": fp += 1
                elif matrix_status == "False Negative (FN)": fn += 1
                
                results.append({
                    "Filename": file.name,
                    "Predicted Class": "Pes Planus (1)" if pred_label == 1 else "Normal (0)",
                    "Confidence Score": f"{confidence:.2f}%" if confidence is not None else "N/A",
                    "Actual Ground Truth": "Pes Planus (1)" if actual_label == 1 else ("Normal (0)" if actual_label == 0 else "Unspecified"),
                    "Matrix Evaluation": matrix_status
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.success("✅ ประมวลผลภาพทั้งหมดเสร็จสิ้น!")
            
            # Save results to DataFrame
            df_results = pd.DataFrame(results)
            
            # Display Evaluation Metrics if CSV was provided
            if ground_truth_dict:
                st.markdown("### 🎯 ตาราง Confusion Matrix Evaluation")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("True Positive (TP)", tp, help="ทายว่าเท้าแบน และตรงกับเฉลย")
                m2.metric("True Negative (TN)", tn, help="ทายว่าปกติ และตรงกับเฉลย")
                m3.metric("False Positive (FP)", fp, help="ทายว่าเท้าแบน ทั้งที่จริงปกติ (Error)")
                m4.metric("False Negative (FN)", fn, help="ทายว่าปกติ ทั้งที่จริงเท้าแบน (Error/หลุดตรวจ)")
                
                # Advanced Performance Indicators
                total_evaluated = tp + tn + fp + fn
                if total_evaluated > 0:
                    acc = (tp + tn) / total_evaluated * 100
                    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
                    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
                    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0
                    
                    st.markdown("#### 📈 ค่าดรรชนีวัดประสิทธิภาพการทำนาย")
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Accuracy", f"{acc:.2f}%")
                    p2.metric("Precision", f"{precision:.2f}%")
                    p3.metric("Sensitivity (Recall)", f"{recall:.2f}%")
                    p4.metric("Specificity", f"{specificity:.2f}%")
            
            # Display Table Summary
            st.markdown("---")
            st.markdown("### 📋 ตารางสรุปผลการวิเคราะห์รายภาพ")
            st.dataframe(df_results, use_container_width=True)
            
            # CSV Download Button
            csv_data = df_results.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 ดาวน์โหลดผลการวิเคราะห์เป็น CSV",
                data=csv_data,
                file_name="pes_planus_diagnosis_results.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("👨‍⚕️ *หมายเหตุ: ระบบ AI นี้จัดทำขึ้นเพื่อช่วยสนับสนุนการวินิจฉัยเบื้องต้น การวินิจฉัยขั้นสุทธิควรได้รับการยืนยันโดยแพทย์ผู้เชี่ยวชาญด้านออร์โธปิดิกส์/รังสีแพทย์*")