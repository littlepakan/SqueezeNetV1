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
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.markdown('<div class="main-header">🦶 ระบบวินิจฉัยภาวะเท้าแบนด้วย AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Diagnosis of Pes Planus via Calcaneal Inclusion Angle Feature Extraction</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.header("🧠 1. เลือกโมเดลสกัดลักษณะเด่น")
dl_model_choice = st.sidebar.radio(
    "Deep Learning Feature Extractor:",
    ("SqueezeNet", "GoogLeNet")
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. การตั้งค่าโมเดล Machine Learning (SVM & RFE)")
st.sidebar.warning(f"⚠️ กรุณาเลือกไฟล์ SVM และ RFE ที่ถูกเทรนมาสำหรับ **{dl_model_choice}** เท่านั้น")

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
        st.sidebar.warning("⚠️ ไม่พบไฟล์ .pkl ในโฟลเดอร์ โปรดวางไฟล์ Fold_X_best_svm.pkl ในไดเรกทอรีเดียวกัน")

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
# Load Deep Learning Model (Cached for Speed)
# ---------------------------------------------------------
@st.cache_resource
def load_feature_extractor(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "SqueezeNet":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    elif model_name == "GoogLeNet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        # นำ Fully Connected layer ออกเพื่อดึงแค่ Feature (ได้ 1024 มิติ)
        model.fc = nn.Identity()
        
    model = model.to(device)
    model.eval()
    return model, device

feature_extractor, device = load_feature_extractor(dl_model_choice)

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
# Main Interface: File Upload & Evaluation
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 1. อัปโหลดภาพเอกซเรย์เท้า (Multiple Images)")
    uploaded_files = st.file_uploader(
        "เลือกไฟล์ภาพ... (อัปโหลดได้หลายไฟล์พร้อมกัน)", 
        type=["jpg", "jpeg", "png", "bmp"], 
        accept_multiple_files=True
    )

with col2:
    st.subheader("📄 2. อัปโหลดไฟล์เฉลย (Optional)")
    st.info("ไฟล์ .csv ต้องมีคอลัมน์ `filename` (ชื่อไฟล์รวมนามสกุล) และ `label` (0 = ปกติ, 1 = เท้าแบน)")
    csv_file = st.file_uploader("เลือกไฟล์ CSV...", type=["csv"])

st.markdown("---")

if uploaded_files:
    if svm_model is None or rfe_selector is None:
        st.error("⚠️ กรุณาโหลดไฟล์โมเดล (SVM/RFE) ที่แถบด้านซ้ายก่อนทำการวิเคราะห์")
    else:
        if st.button("🚀 เริ่มประมวลผลและวินิจฉัยทั้งหมด", type="primary", use_container_width=True):
            
            # --- Load Ground Truth from CSV ---
            ground_truth = {}
            if csv_file is not None:
                try:
                    df_gt = pd.read_csv(csv_file)
                    if 'filename' in df_gt.columns and 'label' in df_gt.columns:
                        df_gt['filename'] = df_gt['filename'].astype(str)
                        ground_truth = dict(zip(df_gt['filename'], df_gt['label']))
                        st.success(f"โหลดข้อมูลเฉลยสำเร็จ จำนวน {len(ground_truth)} รายการ")
                    else:
                        st.error("❌ ไฟล์ CSV ไม่ถูกต้อง: ไม่พบคอลัมน์ 'filename' หรือ 'label'")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")

            # --- Processing Images ---
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"⏳ กำลังประมวลผล: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    img_tensor = eval_transforms(image).unsqueeze(0).to(device)
                    
                    # 1. Extract Features
                    with torch.no_grad():
                        features = feature_extractor(img_tensor).cpu().numpy()
                    
                    # 2. Apply RFE & SVM Predict
                    features_opt = rfe_selector.transform(features)
                    prediction = svm_model.predict(features_opt)[0]
                    
                    # 3. Check against Ground Truth
                    filename = uploaded_file.name
                    gt_label = ground_truth.get(filename, None)
                    
                    eval_status = "ไม่มีเฉลย"
                    if gt_label is not None:
                        if gt_label == 1 and prediction == 1:
                            eval_status = "True Positive (TP)"
                        elif gt_label == 0 and prediction == 0:
                            eval_status = "True Negative (TN)"
                        elif gt_label == 0 and prediction == 1:
                            eval_status = "False Positive (FP)"
                        elif gt_label == 1 and prediction == 0:
                            eval_status = "False Negative (FN)"

                    results.append({
                        "Filename": filename,
                        "Prediction": "Pes Planus (1)" if prediction == 1 else "Normal (0)",
                        "Ground Truth": f"Pes Planus (1)" if gt_label == 1 else (f"Normal (0)" if gt_label == 0 else "-"),
                        "Evaluation": eval_status
                    })
                
                except ValueError as ve:
                    st.error(f"เกิดข้อผิดพลาดกับภาพ {uploaded_file.name}: ขนาด Feature ไม่ตรงกัน โปรดตรวจสอบว่าโมเดล SVM ฝึกมากับ {dl_model_choice} หรือไม่ (รายละเอียด: {ve})")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดกับภาพ {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ ประมวลผลเสร็จสิ้น")
            
            # --- Display Results ---
            if results:
                st.write("### 📊 ตารางสรุปผลการวินิจฉัย")
                df_results = pd.DataFrame(results)
                
                # ฟังก์ชันไฮไลท์สีในตาราง
                def highlight_eval(val):
                    if 'TP' in val or 'TN' in val:
                        return 'background-color: #D1FAE5; color: #065F46; font-weight: bold;'
                    elif 'FP' in val or 'FN' in val:
                        return 'background-color: #FEE2E2; color: #7F1D1D; font-weight: bold;'
                    return ''

                # ใช้ Pandas Styler ในการตกแต่งตาราง
                styled_df = df_results.style.map(highlight_eval, subset=['Evaluation'])
                st.dataframe(styled_df, use_container_width=True)

                # --- Calculate and Show Metrics if Ground Truth is provided ---
                if len(ground_truth) > 0:
                    st.write("### 🎯 สรุปประสิทธิภาพ (Confusion Matrix Metrics)")
                    
                    tp = sum(1 for r in results if 'TP' in r['Evaluation'])
                    tn = sum(1 for r in results if 'TN' in r['Evaluation'])
                    fp = sum(1 for r in results if 'FP' in r['Evaluation'])
                    fn = sum(1 for r in results if 'FN' in r['Evaluation'])
                    total_evaluated = tp + tn + fp + fn
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.markdown(f"<div class='metric-card'><h3 style='color: #059669;'>TP: {tp}</h3><p>True Positive</p></div>", unsafe_allow_html=True)
                    m2.markdown(f"<div class='metric-card'><h3 style='color: #059669;'>TN: {tn}</h3><p>True Negative</p></div>", unsafe_allow_html=True)
                    m3.markdown(f"<div class='metric-card'><h3 style='color: #DC2626;'>FP: {fp}</h3><p>False Positive</p></div>", unsafe_allow_html=True)
                    m4.markdown(f"<div class='metric-card'><h3 style='color: #DC2626;'>FN: {fn}</h3><p>False Negative</p></div>", unsafe_allow_html=True)
                    
                    if total_evaluated > 0:
                        accuracy = ((tp + tn) / total_evaluated) * 100
                        st.markdown(f"""
                        <div style='background-color: #EFF6FF; padding: 1.5rem; border-radius: 0.5rem; text-align: center; margin-top: 1.5rem; border: 1px solid #BFDBFE;'>
                            <h2 style='color: #1D4ED8; margin: 0;'>ความแม่นยำรวม (Accuracy): {accuracy:.2f}%</h2>
                            <p style='margin: 0; color: #3B82F6;'>จากจำนวนภาพที่มีเฉลยทั้งหมด {total_evaluated} ภาพ</p>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("👈 กรุณาอัปโหลดภาพเอกซเรย์อย่างน้อย 1 ภาพเพื่อเริ่มการทำงาน")

st.markdown("---")
st.caption("👨‍⚕️ *หมายเหตุ: ระบบนี้ออกแบบมาเพื่อสนับสนุนการวิจัยและทดสอบประสิทธิภาพโมเดล Deep Learning ควบคู่กับ Machine Learning*")