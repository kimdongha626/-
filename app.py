import streamlit as st

# 1. 페이지 설정 (웹 브라우저 탭 타이틀 및 아이콘)
st.set_page_config(page_title="당뇨 예측 프로그램", page_icon="🩺", layout="centered")

# 2. 타이틀 및 설명 영역
st.title("🩺 당뇨 예측 프로그램")
st.write("생활 정보를 입력하면 당뇨 여부를 예측합니다.")
st.markdown("---") # 구분선

# 3. 2단 레이아웃 (좌측 열, 우측 열) 구성
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("임신횟수", min_value=0, value=0, step=1)
    glucose = st.number_input("혈당", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    blood_pressure = st.number_input("혈압", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    skin_thickness = st.number_input("피부두께", min_value=0.0, value=0.0, step=1.0, format="%.2f")

with col2:
    insulin = st.number_input("인슐린", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    bmi = st.number_input("체질량지수(BMI)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    diabetes_pedigree = st.number_input("가족력", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    age = st.number_input("나이", min_value=0, value=0, step=1)

st.markdown("---") # 구분선

# 4. 예측하기 버튼 (가로로 꽉 차는 스타일 적용)
# Streamlit의 use_container_width=True 옵션으로 이미지처럼 긴 버튼을 만듭니다.
predict_button = st.button("🔍 당뇨 예측하기", use_container_width=True)

# 5. 버튼 클릭 시 동작 (이곳에 모델 예측 로직을 연결하시면 됩니다)
if predict_button:
    # 입력된 데이터 모으기
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }
    
    # 예시 결과 출력 (추후 인공지능 모델 결과로 대체 가능)
    st.success("데이터가 성공적으로 입력되었습니다! 예측 로직을 연결해 주세요.")
    st.write("입력된 데이터 확인:", input_data)
