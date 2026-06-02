import streamlit as st
import pandas as pd
import numpy as np

# [주의] 기존 코드에서 사용하신 모델과 스케일러, 학습 데이터 컬럼(X) 정보가 미리 로드되어 있어야 합니다.
# 예시: rf_model_eng = joblib.load('model.pkl')
# 예시: scaler = joblib.load('scaler.pkl')
# 예시: X_columns = ['임신횟수', '혈당', '혈압', '피부 두께', '인슐린', 'BMI', '가족력', '나이', 'Glucose_Insulin_Ratio', ...]

# 1. 페이지 설정
st.set_page_config(page_title="당뇨 예측 프로그램", page_icon="🩺", layout="centered")

# 2. 타이틀 및 설명
st.title("🩺 당뇨 예측 프로그램")
st.write("생활 정보를 입력하면 당뇨 여부를 예측합니다.")
st.markdown("---")

# 3. 2단 레이아웃으로 입력창 배치 (보내주신 이미지 UI 반영)
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

# 4. 당뇨 예측하기 버튼
predict_button = st.button("🔍 당뇨 예측하기", use_container_width=True)

# 5. 버튼 클릭 시 데이터 처리 및 예측
if predict_button:
    
    # [데이터프레임 변환]
    input_data_original = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=['임신횟수', '혈당', '혈압', '피부 두께', '인슐린', 'BMI', '가족력', '나이']
    )
    
    # [파생 변수 자동 계산]
    input_data_engineered = input_data_original.copy()
    
    # 1. 포도당-인슐린 비율
    input_data_engineered['Glucose_Insulin_Ratio'] = input_data_engineered['혈당'] / input_data_engineered['인슐린'].replace(0, 1)
    
    # 2. BMI와 나이 상호작용
    input_data_engineered['BMI_Age_Interaction'] = input_data_engineered['BMI'] * input_data_engineered['나이']
    
    # 3. 고혈압 지표
    input_data_engineered['High_BloodPressure'] = (input_data_engineered['혈압'] >= 80).astype(int)
    
    # 4. 고령층 여부
    input_data_engineered['Elderly'] = (input_data_engineered['나이'] >= 50).astype(int)
    
    # 5. 복합 건강 위험 점수
    input_data_engineered['Composite_Risk_Score'] = (
        input_data_engineered['임신횟수'] + 
        (input_data_engineered['혈당'] / 100) + 
        (input_data_engineered['BMI'] / 10) + 
        (input_data_engineered['나이'] / 10)
    )
    
    try:
        # [컬럼 순서 정렬] 기존 X.columns 대입 (만약 X 객체가 없다면 학습할 때 쓴 컬럼 리스트 문자열을 직접 넣으셔도 됩니다)
        input_data_final = input_data_engineered[X.columns]
        
        # [예측 진행]
        input_scaled = scaler.transform(input_data_final)
        predicted = rf_model_eng.predict(input_scaled)
        prob = rf_model_eng.predict_proba(input_scaled)
        
        # [결과 화면 출력]
        st.subheader("📊 예측 결과")
        
        # 당뇨 확률 계산
        diabetes_probability = prob[0][1] * 100
        
        if predicted[0] == 1:
            st.error(f"⚠️ 당뇨병 위험군으로 예측됩니다. (당뇨 확률: **{diabetes_probability:.1f}%**)")
        else:
            st.success(f"✅ 정상군으로 예측됩니다. (당뇨 확률: **{diabetes_probability:.1f}%**)")
            
        # 게이지 바 형태로 시각화 추가 (선택 사항)
        st.progress(int(diabetes_probability))
        
    except NameError as e:
        st.error("💡 오류: 모델(`rf_model_eng`), 스케일러(`scaler`), 혹은 훈련 데이터 컬럼 정보(`X`)가 코드 상단에 로드되지 않았습니다. 파일 상단에 모델을 불러오는 코드를 추가해 주세요.")
