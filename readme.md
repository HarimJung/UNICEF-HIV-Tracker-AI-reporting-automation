# 1-1. 현재 활성화된 가상 환경 비활성화
deactivate 

# 1-2. 기존 venv 폴더 (충돌 가능성이 있는) 완전히 삭제
# (주의: 'venv'가 프로젝트 폴더에 있을 경우에만 사용)
rm -rf venv

# 2-1. 새로운 가상 환경 생성
python3 -m venv venv

# 2-2. 새로운 가상 환경 활성화
source venv/bin/activate

# 3-1. 모든 필수 라이브러리 일괄 설치
pip install -r requirements.txt

# 터미널에서 실행
python create_hiv_data.py

streamlit un app.py