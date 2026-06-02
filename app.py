import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(page_title="당뇨 예측 프로그램", page_icon="🩺", layout="centered")


# ==========================================
# 2. 모델 및 스케일러 로드 함수
# ==========================================
@st.cache_resource
def load_models():
    # 같은 폴더에 저장된 파일들을 읽어옵니다.
    model = joblib.load("rf_model_eng.pkl") 
    scaler_obj = joblib.load("scaler.pkl")
    return model, scaler_obj

try:
    rf_model_eng, scaler = load_models()
    model_loaded = True
except FileNotFoundError:
    st.error("🚨 `rf_model_eng.pkl` 또는 `scaler.pkl` 파일을 찾을 수 없습니다. 모델 파일을 깃허브 최상위 폴더에 함께 올려주세요.")
    model_loaded = False


# ==========================================
# 3. 훈련 데이터 컬럼 순서 정의
# ==========================================
class TrainingColumns:
    columns = [
        '임신횟수', '혈당', '혈압', '피부 두께', '인슐린', 'BMI', '가족력', '나이',
        'Glucose_Insulin_Ratio', 'BMI_Age_Interaction', 'High_BloodPressure', 'Elderly', 'Composite_Risk_Score'
    ]

X = TrainingColumns()


# ==========================================
# 4. UI 디자인 및 입력창 (2단 구성)
# ==========================================
st.title("🩺 당뇨 예측 프로그램")
st.write("생활 정보를 입력하면 당뇨 여부를 예측합니다.")
st.markdown("---")

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
# 5. 예측 실행 로직
# ==========================================
predict_button = st.button("🔍 당뇨 예측하기", use_container_width=True)

if predict_button:
    if not model_loaded:
        st.warning("모델 파일이 로드되지 않아 예측을 수행할 수 없습니다.")
    else:
        # 데이터프레임 변환
        input_data_original = pd.DataFrame(
            [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=['임신횟수', '혈당', '혈압', '피부 두께', '인슐린', 'BMI', '가족력', '나이']
        )
        
        # 파생 변수 계산
        input_data_engineered = input_data_original.copy()
        input_data_engineered['Glucose_Insulin_Ratio'] = input_data_engineered['혈당'] / input_data_engineered['인슐린'].replace(0, 1)
        input_data_engineered['BMI_Age_Interaction'] = input_data_engineered['BMI'] * input_data_engineered['나이']
        input_data_engineered['High_BloodPressure'] = (input_data_engineered['혈압'] >= 80).astype(int)
        input_data_engineered['Elderly'] = (input_data_engineered['나이'] >= 50).astype(int)
        input_data_engineered['Composite_Risk_Score'] = (
            input_data_engineered['임신횟수'] + 
            (input_data_engineered['혈당'] / 100) + 
            (input_data_engineered['BMI'] / 10) + 
            (input_data_engineered['나이'] / 10)
        )
        
        # 컬럼 순서 정렬 및 스케일링, 예측
        input_data_final = input_data_engineered[X.columns]
        input_scaled = scaler.transform(input_data_final)
        
        predicted = rf_model_eng.predict(input_scaled)
        prob = rf_model_eng.predict_proba(input_scaled)
        diabetes_probability = prob[0][1] * 100
        
        # 결과 출력
        st.subheader("📊 예측 결과")
        if predicted[0] == 1:
            st.error(f"⚠️ 당뇨병 위험군으로 예측됩니다. (당뇨 확률: **{diabetes_probability:.1f}%**)")
        else:
            st.success(f"✅ 정상군으로 예측됩니다. (당뇨 확률: **{diabetes_probability:.1f}%**)")
            
        st.progress(int(diabetes_probability))
