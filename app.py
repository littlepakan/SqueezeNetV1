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
import xgboost  # จำเป็นต้องมีเพื่อให้ Pickle สามารถโหลดไฟล์ XGBoost ได้
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==========================================
# Set Streamlit Page Configuration
# ==========================================
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

# ==========================================
# Sidebar Configuration
# ==========================================
st.sidebar.header("🧠 1. เลือกโครงข่ายประสาทเทียม (Backbone)")
dl_model_choice = st.sidebar.radio(
    "Deep Learning Model:",
    ("SqueezeNet", "GoogleNet")
)

st.sidebar.header("⚙️ 2. เลือกตัวจำแนกโรค (Classifier)")
classifier_choice = st.sidebar.radio(
    "Classifier Mode:",
    ("SVM", "RandomForest", "XGBoost", "NeuralNetwork", "Fine-Tuning")
)

st.sidebar.markdown("---")
is_finetuned = (classifier_choice == "Fine-Tuning")

st.sidebar.header("📂 3. การอัปโหลดโมเดล")
st.sidebar.warning(f"⚠️ กรุณาอัปโหลดไฟล์ที่ฝึกสอนคู่กับ **{dl_model_choice}** และ **{classifier_choice}** เท่านั้น")

# ตัวแปรเก็บโมเดล
ml_model = None
rfe_selector = None
finetuned_weights_path = None

if is_finetuned:
    # โหมด Fine-Tuning ต้องการแค่ไฟล์ .pth
    uploaded_pth = st.sidebar.file_uploader(f"อัปโหลดไฟล์ Weights (.pth) สำหรับ {dl_model_choice}", type=["pth"], key="pth")
    if uploaded_pth:
        # บันทึกไฟล์ชั่วคราวเพื่อให้ PyTorch โหลดได้
        with open("temp_model.pth", "wb") as f:
            f.write(uploaded_pth.getbuffer())
        finetuned_weights_path = "temp_model.pth"
        st.sidebar.success("โหลดไฟล์ .pth สำเร็จ!")
else:
    # โหมด Machine Learning ปกติ ต้องการ .pkl ของ Classifier และ RFE
    uploaded_clf = st.sidebar.file_uploader(f"อัปโหลดไฟล์ {classifier_choice} Model (.pkl)", type=["pkl"], key="clf")
    uploaded_rfe = st.sidebar.file_uploader(f"อัปโหลดไฟล์ RFE Selector (.pkl)", type=["pkl"], key="rfe")
    
    if uploaded_clf and uploaded_rfe:
        try:
            ml_model = pickle.load(uploaded_clf)
            rfe_selector = pickle.load(uploaded_rfe)
            st.sidebar.success(f"โหลดไฟล์ {classifier_choice} สำเร็จ!")
        except Exception as e:
            st.sidebar.error(f"ไฟล์โมเดลไม่ถูกต้อง: {e}")

# ==========================================
# Load Deep Learning Model Function
# ==========================================
@st.cache_resource
def load_pytorch_model(model_name, is_ft):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "SqueezeNet":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT if not is_ft else None)
        if is_ft:
            # โครงสร้างสำหรับ Fine-Tuned SqueezeNet
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, 2, kernel_size=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        else:
            # โครงสร้างสำหรับ Feature Extractor
            model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            
    elif model_name == "GoogleNet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT if not is_ft else None)
        model.aux_logits = False
        if is_ft:
            # โครงสร้างสำหรับ Fine-Tuned GoogleNet
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(1024, 2)
            )
        else:
            # โครงสร้างสำหรับ Feature Extractor
            model.fc = nn.Identity()
            
    model = model.to(device)
    model.eval()
    return model, device

# ==========================================
# Image Preprocessing Setup
# ==========================================
def apply_median_filter(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    median_img = cv2.medianBlur(img_cv, 3)
    final_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_img)

# กำหนดขนาดภาพตามโมเดล
IMG_SIZE = 227 if dl_model_choice == "SqueezeNet" else 224

eval_transforms = transforms.Compose([
    transforms.Lambda(apply_median_filter),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# Main Interface: File Upload & Evaluation
# ==========================================
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
    st.info("ไฟล์ .csv ต้องมีคอลัมน์ `img_name` (ชื่อรูป) และ `label` (0 = ปกติ, 1 = เท้าแบน)")
    csv_file = st.file_uploader("เลือกไฟล์ CSV...", type=["csv"])

st.markdown("---")

# เช็กความพร้อมของระบบ
is_ready_to_predict = False
if is_finetuned and finetuned_weights_path is not None:
    is_ready_to_predict = True
elif not is_finetuned and ml_model is not None and rfe_selector is not None:
    is_ready_to_predict = True

if uploaded_files:
    if not is_ready_to_predict:
        st.error(f"⚠️ กรุณาอัปโหลดไฟล์โมเดลที่จำเป็นสำหรับโหมด **{classifier_choice}** ที่แถบด้านซ้ายมือให้ครบถ้วนก่อนทำการวิเคราะห์")
    else:
        if st.button("🚀 เริ่มประมวลผลและวินิจฉัยทั้งหมด", type="primary", use_container_width=True):
            
            # --- Load Deep Learning Model ---
            dl_model, device = load_pytorch_model(dl_model_choice, is_finetuned)
            
            if is_finetuned:
                # โหลด Weights ที่ผู้ใช้อัปโหลดใส่เข้าไปในโมเดล
                dl_model.load_state_dict(torch.load(finetuned_weights_path, map_location=device))
                dl_model.eval()

            # --- Load Ground Truth from CSV ---
            ground_truth = {}
            if csv_file is not None:
                try:
                    df_gt = pd.read_csv(csv_file)
                    if 'img_name' in df_gt.columns and 'label' in df_gt.columns:
                        for _, row in df_gt.iterrows():
                            # ดึงชื่อไฟล์และจำทั้งแบบมีนามสกุลและไม่มีนามสกุล (เพื่อความทนทาน)
                            raw_name = str(row['img_name']).strip()
                            base_name = os.path.splitext(raw_name)[0]
                            lbl = int(row['label'])
                            ground_truth[raw_name] = lbl
                            ground_truth[base_name] = lbl
                        st.success(f"โหลดข้อมูลเฉลยสำเร็จ จำนวนรายการอ้างอิง: {len(df_gt)} รายการ")
                    else:
                        st.error("❌ ไฟล์ CSV ไม่ถูกต้อง: ไม่พบคอลัมน์ 'img_name' หรือ 'label'")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")

            # --- Processing Images ---
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                base_filename = os.path.splitext(filename)[0]
                status_text.text(f"⏳ กำลังประมวลผล: {filename} ({i+1}/{len(uploaded_files)})")
                
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    img_tensor = eval_transforms(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        if is_finetuned:
                            # 🌟 โหมด Fine-Tuning: โมเดลทายผลออกมาโดยตรง
                            outputs = dl_model(img_tensor)
                            _, preds = torch.max(outputs.data, 1)
                            prediction = preds.item()
                        else:
                            # 🌟 โหมด Machine Learning: สกัด Feature -> RFE -> Predict
                            features = dl_model(img_tensor).cpu().numpy()
                            features_opt = rfe_selector.transform(features)
                            prediction = ml_model.predict(features_opt)[0]
                    
                    # 3. Check against Ground Truth
                    # พยายามค้นหาชื่อไฟล์แบบเต็มก่อน ถ้าไม่เจอให้หาแบบตัดนามสกุลออก
                    gt_label = ground_truth.get(filename, ground_truth.get(base_filename, None))
                    
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
                
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดกับภาพ {filename}: {e}")
                
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
                            <p style='margin: 0; color: #3B82F6;'>จากจำนวนภาพที่มีเฉลยทั้งหมด {total_evaluated} ภาพ | ทดสอบด้วย {dl_model_choice} + {classifier_choice}</p>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("👈 กรุณาอัปโหลดภาพเอกซเรย์อย่างน้อย 1 ภาพเพื่อเริ่มการทำงาน")

st.markdown("---")
st.caption("👨‍⚕️ *หมายเหตุ: ระบบนี้ออกแบบมาเพื่อสนับสนุนการวิจัยและทดสอบประสิทธิภาพโมเดล Deep Learning ควบคู่กับ Machine Learning*")