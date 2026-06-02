import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. 페이지 설정 (브라우저 탭 타이틀 및 아이콘)
# ==========================================
st.set_page_config(page_title="당뇨 예측 프로그램", page_icon="🩺", layout="centered")


# ==========================================
# 2. 모델 및 스케일러 로드 함수
# ==========================================
@st.cache_resource  # 앱이 실행될 때마다 모델을 매번 새로 로드하지 않도록 캐싱
def load_models():
    # 같은 폴더에 있는 모델과 스케일러 파일을 읽어옵니다.
    # 파일명이 다르거나 경로가 다르면 이 부분을 수정해 주세요.
    model = joblib.load("diabetes.pkl") 
    scaler_obj = joblib.load("scaler.pkl")
    return model, scaler_obj

try:
    rf_model_eng, scaler = load_models()
    model_loaded = True
except FileNotFoundError:
    st.error("🚨 `diabetes.pkl` 또는 `scaler.pkl` 파일을 찾을 수 없습니다. 모델 파일을 `app.py`와 같은 폴더에 넣어주세요.")
    model_loaded = False


# ==========================================
# 3. 훈련 데이터 컬럼 순서 정의 (X.columns 대용)
# ==========================================
class TrainingColumns:
    columns = [
        '임신횟수', '혈당', '혈압', '피부 두께', '인슐린', 'BMI', '가족력', '나이',
        'Glucose_Insulin_Ratio', 'BMI_Age_Interaction', 'High_BloodPressure', 'Elderly', 'Composite_Risk_Score'
    ]

X = TrainingColumns()


# ==========================================
# 4. 스트림릿 UI 디자인 (메인 타이틀)
# ==========================================
st.title("🩺 당뇨 예측 프로그램")
st.write("생활 정보를 입력하면 당뇨 여부를 예측합니다.")
st.markdown("---")


# ==========================================
# 5. 2단 입력 레이아웃 구성 (이미지 UI 반영)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("임신횟수", min_value=0, value=0, step=1)
    glucose = st.number_input("혈당", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    bp = st.number_input("혈압", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    skin = st.number_input("피부두께", min_value=0.0, value=0.0, step=1.0, format="%.2f")

with col2:
    insulin = st.number_input("인슐린", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    bmi = st.number_input("체질량지수(BMI)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    dpf = st.number_input("가족력", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    age = st.number_input("나이", min_value=0, value=0, step=1)

st.markdown("---")


# ==========================================
# 6. 예측 실행 로직
# ==========================================
predict_button = st.button("🔍 당뇨 예측하기", use_container_width=True)

if predict_button:
    if not model_loaded:
        st.warning("모델 파일이 로드되지 않아 예측을 수행할 수 없습니다.")
    else:
        # [DataFrame 변환] 입력창의 데이터 수집
        input_data_original = pd.DataFrame(
            [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=['임신횟수', '혈당', '혈압', '피부 두께', '인슐린', 'BMI', '가족력', '나이']
        )
        
        # [파생 변수 자동 계산]
        input_data_engineered = input_data_original.copy()
