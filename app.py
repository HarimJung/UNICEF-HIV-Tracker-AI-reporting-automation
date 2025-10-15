import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import os 
import numpy as np
# URL 인코딩을 위해 quote_plus 함수를 임포트합니다.
from urllib.parse import quote_plus 

# =================================================================
# 🚨 [최상단 설정] 페이지 설정은 Streamlit 명령 중 반드시 최상단에 위치해야 합니다.
# =================================================================
st.set_page_config(layout="wide") 

# =================================================================
# [필수 설정] 1. API 키, 파일 및 Tableau 정보 입력
# =================================================================

# 1. 사용자 제공 API 키 반영 (새로운 키로 업데이트됨)
# 🚨 이 키는 새로운 키입니다.
OPENAI_API_KEY = "yourAPIKey".strip()

# 2. 사용자 제공 Tableau URL 반영 (HIV 대시보드 URL)
TABLEAU_BASE_URL = "https://public.tableau.com/views/UNICEFHIVTech_AI/UNICEFHIVReporting?:showVizHome=no&:embed=true"

# 3. 🚨 Tableau 파라미터 이름 설정 (표준 쿼리 스트링 이름 사용)
TABLEAU_FILTER_FIELD_COUNTRY = "Country"         # Tableau Filter Name: Country
TABLEAU_FILTER_FIELD_YEAR = "Year"               # Tableau Filter Name: Year
TABLEAU_FILTER_FIELD_REGION = "Unicef Region"    # Tableau Filter Name: Unicef Region (공백 포함)

# 데이터 파일명 설정
DATA_FILENAME = "unicef_hiv_tech.csv"

# =================================================================
# [데이터 로드 및 최적화]
# =================================================================
@st.cache_data
def load_and_prepare_data(filename):
    """CSV 파일을 로드하고 AI 분석에 필요한 데이터를 최적화하여 반환합니다."""
    try:
        # 파일 로드 (인코딩 문제 발생 시 'latin-1' 등으로 변경 가능)
        df = pd.read_csv(filename, encoding='utf-8') 
        
        # 1. 필수 열 결측치 제거: PLHIV, ART, 신규 감염 등 핵심 지표가 있는 행만 분석에 사용
        df_filtered = df.dropna(subset=['country', 'year', 'PLHIV_0_19', 'ART_Coverage_0_14_Pct', 'Annual_New_Infections_0_14']).copy()

        # 2. Year를 정수형으로 변환
        df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce').astype('Int64')
        
        # 3. 최근 5년 데이터만 추출하여 LLM 프롬프트에 제공
        if not df_filtered.empty:
            latest_year = df_filtered['year'].max()
            start_year = latest_year - 4
            df_sample = df_filtered[df_filtered['year'] >= start_year].copy()
            
            # 4. PLHIV 수가 많은 상위 30개국으로 필터링하여 LLM의 분석 효율성 증대
            top_30_countries = df_sample.groupby('country')['PLHIV_0_19'].max().nlargest(30).index.tolist()
            df_sample = df_sample[df_sample['country'].isin(top_30_countries)]
        else:
            df_sample = df_filtered
        
        return df_sample
    
    except FileNotFoundError:
        st.error(f"🚨 파일을 찾을 수 없습니다: {filename} 파일을 VS Code 작업 폴더에 넣어주세요.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 데이터 로드 중 오류 발생: {e}")
        st.stop()

df_sample = load_and_prepare_data(DATA_FILENAME)

# OpenAI 클라이언트 초기화 및 키 유효성 검사
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.error("🚨 OpenAI API 키가 유효하지 않거나 설정되지 않았습니다.")
    st.stop()
    client = None 

# =================================================================
# [핵심 기능 1 & 3] LLM 호출 및 답변 생성 함수
# =================================================================

def generate_ai_response(user_question, df_sample):
    """LLM을 호출하여 데이터 분석 및 보고서 초안 생성을 요청합니다."""
    if not client: return "API 클라이언트 오류로 분석을 실행할 수 없습니다. 키를 확인하세요."
    
    # LLM에게 제공할 데이터 요약 (주요 지표만 선택)
    data_summary = df_sample[[
        'country', 'year', 'unicef_region', 'PLHIV_0_19', 'ART_Coverage_0_14_Pct', 
        'Annual_New_Infections_0_14', 'Annual_AIDS_Deaths_0_14', 'MTCT_Rate_Pct'
    ]].to_markdown(index=False)

    # 기본값 설정
    latest_year_in_data = str(df_sample['year'].max() if not df_sample.empty else 2024)
    default_country_name = 'South Africa' 
    
    # 🌟 프롬프트 수정: UNICEF HIV 데이터 및 질문에 맞게 변경 🌟
    system_prompt = f"""
    당신은 **UNICEF 소속의 AI 기반 HIV 정책 컨설턴트**입니다.
    주어진 CSV 데이터(최근 5년간 PLHIV가 많은 상위 30개국의 데이터)를 바탕으로 사용자의 질문에 답변하고, 정책 보고서 초안을 작성하세요.
    
    ### [분석 데이터 요약 (최근 5년 상위 30개국)]
    {data_summary}
    
    ### [분석 지침 및 데이터 정의]
    1.  **데이터 기반:** 답변은 반드시 위 표의 실제 데이터를 기반으로 사실을 제시해야 합니다.
    2.  **선정 및 기본값 강제:** 분석 후 Tableau 대시보드 제어를 위해 **반드시** 최종 국가 1개와 연도 1개를 선정해야 합니다. 질문의 답을 찾지 못하더라도, **기본값**으로 '{default_country_name}'와 '{latest_year_in_data}'를 선정해야 합니다.
    3.  **보고서 초안:** 선정된 **단일 국가**에 초점을 맞추어 작성하며, PLHIV_0_19, ART_Coverage_0_14_Pct, Annual_New_Infections_0_14 필드를 활용해야 합니다.

    ### 🚨🚨🚨 [최종 출력 태그 포맷 절대적 강제] 🚨🚨🚨
    1.  모든 답변 내용(분석 보고서 포함)이 모두 끝난 후, **맨 마지막 줄에 단독으로 한 줄**만 사용해야 합니다.
    2.  **정확한 포맷:** [FILTER_COUNTRY: [국가 영어 이름]][FILTER_YEAR: [연도 숫자]]
    3.  **국가 이름:** 데이터셋에 존재하는 **영어 국가 이름**만 사용하세요.
    4.  **예시 (절대로 이 포맷에서 벗어나지 마세요):** [FILTER_COUNTRY: {default_country_name}][FILTER_YEAR: {latest_year_in_data}]
    5.  **경고:** 이 태그 외에 다른 텍스트, 공백, 줄 바꿈, 설명 등을 마지막 줄에 절대 추가하지 마세요. 오직 태그 문자열만 포함해야 합니다.
    """
    
    # 사용자 질문 및 보고서 초안 요청 통합
    prompt_with_report = f"""
    [사용자 질문]: {user_question}
    
    분석 답변을 먼저 제공한 후, 아래 Powerpoint 슬라이드 형식에 맞춰 보고서 초안을 작성해주세요.
    
    [보고서 초안] (선정된 단일 국가에 초점을 맞춰 작성)
    1. 슬라이드 1. [제목: 선정 국가 HIV 대응 현황]: 질문에 따라 선정된 **단일 국가**의 ART 치료율(`ART_Coverage_0_14_Pct`)과 신규 감염 수(`Annual_New_Infections_0_14`) 추이를 데이터(`PLHIV_0_19` 활용)를 근거로 요약하세요.
    2. 슬라이드 2. [제목: 정책적 개입 권고]: 해당 국가의 MTCT 비율(`MTCT_Rate_Pct`)과 사망률(`Annual_AIDS_Deaths_0_14`) 데이터를 기반으로 UNICEF가 즉시 실행할 수 있는 초기 정책 권고안 2가지.
    
    **✅ 모든 답변과 보고서 작성이 끝나면, 시스템 지침에 따라 맨 마지막 줄에 필터링 태그를 넣어주세요.**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_with_report}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        # 오류 발생 시 API 에러 메시지를 그대로 반환하여 사용자에게 명확히 전달
        return f"API 호출 중 오류가 발생하여 분석을 완료할 수 없습니다. 오류: {e}"

# =================================================================
# [핵심 기능 2] AI 답변에서 구조화된 단일 국가, 연도, 지역 추출 및 URL 생성
# =================================================================

def extract_structured_filter_value(ai_response, tag):
    """구조화된 필터 태그 ([TAG: Value])에서 값을 추출합니다."""
    match = re.search(r'\[' + re.escape(tag) + r':\s*(.*?)\]', ai_response)
    if match:
        return match.group(1).strip()
    return None

def extract_single_country(ai_response, df_sample):
    """AI 답변 텍스트에서 구조화된 국가 이름을 추출합니다."""
    country = extract_structured_filter_value(ai_response, "FILTER_COUNTRY")
    if country and country in df_sample['country'].unique().tolist():
        return country
    return 'South Africa' # 기본값


def extract_single_year(ai_response, df_sample):
    """AI 답변 텍스트에서 구조화된 연도 숫자를 추출합니다."""
    year_str = extract_structured_filter_value(ai_response, "FILTER_YEAR")
    latest_year = str(df_sample["year"].max() if not df_sample.empty else 2024)

    if year_str and year_str.isdigit():
        year = int(year_str)
        if year in df_sample["year"].unique().tolist():
            return year_str
            
    return latest_year # 기본값: 가장 최근 년도


def extract_region_from_country(country_value, df_sample):
    """선택된 국가를 기준으로 UNICEF Region을 데이터프레임에서 조회합니다."""
    if country_value and country_value in df_sample['country'].unique().tolist() and not df_sample.empty:
        # 해당 국가의 가장 최근 데이터를 기준으로 Region을 찾습니다.
        latest_data = df_sample[df_sample['country'] == country_value].sort_values(by='year', ascending=False).iloc[0]
        return str(latest_data['unicef_region'])
    
    return 'Eastern and Southern Africa' # 기본값


def get_filtered_tableau_url(base_url, country_value, year_value, region_value):
    """
    추출된 값을 사용하여 Tableau URL에 표준 쿼리 스트링 형식으로 파라미터를 추가합니다.
    """
    
    # 기존 URL에서 Tableau의 기본 파라미터 (?:...)를 제거하고 순수 URL만 사용
    base_url_clean = base_url.split("?:")[0] 
    
    # 쿼리 리스트 생성 (Tableau 제어 파라미터로 시작)
    query_list = []
    query_list.append(":embed=y") # 임베딩 모드
    query_list.append(":showVizHome=no") # VizHome 숨김
    
    # 필터 파라미터 추가
    filter_items = [
        (TABLEAU_FILTER_FIELD_COUNTRY, country_value), # Country
        (TABLEAU_FILTER_FIELD_YEAR, year_value),       # Year
        (TABLEAU_FILTER_FIELD_REGION, region_value),   # Unicef Region
    ]
    
    for key, value in filter_items:
        if value:
            # 필터 이름(키)과 필터 값(밸류) 모두 URL 인코딩 (공백을 +로)
            encoded_key = quote_plus(key)
            encoded_value = quote_plus(value)
            query_list.append(f"{encoded_key}={encoded_value}")

    # 모든 파라미터를 '&'로 연결
    final_query_string = "&".join(query_list)
    
    # 최종 URL 조합: BASE_URL?param1=value1&param2=value2...
    final_url = f"{base_url_clean}?{final_query_string}"

    return final_url

# =================================================================
# Streamlit UI 구성
# =================================================================

st.title("💡 AI Driven HIV 대응 대시보드 시연 (UNICEF Data)")
st.markdown("### ⚠️ **주의:** Tableau 대시보드는 **표준 쿼리 스트링 방식**으로 세 가지 필터(`Country`, `Year`, `Unicef Region`)를 제어합니다.")
st.markdown("---")

# 1. 대화형 질문 섹션
st.header("1. AI 기반 질문 및 보고서 생성")

# 사용자 요청에 따라 질문 내용 변경 (사용자가 제공한 질문을 기본값으로 사용)
default_question = """
UNICEF의 핵심 우선순위는 'HIV 감염 아동의 치료 보급과 신규 감염 최소화'입니다.
데이터 분석을 통해, 
1. **PLHIV (0-19세) 규모가 크면서도 ART 치료율 (ART_Coverage_0_14_Pct)이 60% 미만**인 국가들을 식별하세요.
2. 이 중, **최근 5년간 신규 감염(Annual_New_Infections_0_14) 감소 추세가 가장 둔화되었거나 증가한 것으로 보이는 단일 국가**를 선정하고, 해당 국가의 대응 실패 요인(MTCT_Rate_Pct 등)을 추론해 주세요.
3. 선정된 **단일 국가**에 초점을 맞추어 정책 보고서 초안을 작성해야 합니다.
"""

user_question = st.text_area(
    "AI HIV 컨설턴트에게 질문하세요:",
    default_question,
    height=200
)

# 세션 상태 초기화 및 관리
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'filtered_country' not in st.session_state:
    st.session_state.filtered_country = ""
if 'filtered_year' not in st.session_state:
    st.session_state.filtered_year = ""
if 'filtered_region' not in st.session_state:
    st.session_state.filtered_region = ""


if st.button("🚀 AI 분석 요청 및 대시보드 필터링", type="primary"):
    if df_sample.empty:
        st.error("🚨 데이터 파일에 유효한 데이터가 부족하여 분석을 진행할 수 없습니다. 파일 내용을 확인하세요.")
        st.stop()
        
    with st.spinner("AI가 UNICEF 데이터를 분석하고, Tableau 파라미터를 업데이트 중입니다..."):
        # 1. LLM 호출 및 답변 생성
        ai_result = generate_ai_response(user_question, df_sample)
        st.session_state.ai_response = ai_result
        
        # 2. 답변에서 단일 국가 추출
        selected_country = extract_single_country(ai_result, df_sample)
        st.session_state.filtered_country = selected_country
        
        # 3. 답변에서 단일 년도 추출
        st.session_state.filtered_year = extract_single_year(ai_result, df_sample)
        
        # 4. 추출된 국가를 기반으로 Region 조회 (데이터프레임 활용)
        st.session_state.filtered_region = extract_region_from_country(selected_country, df_sample)


    if "API 호출 중 오류" not in st.session_state.ai_response:
        st.success("분석 완료! AI의 해석과 자동 파라미터 제어된 대시보드를 확인하세요.")
    else:
        st.error("🚨 분석 실패! API 키와 잔액을 확인해 주세요.")


if st.session_state.ai_response:
    # LLM 답변 출력
    st.markdown("---")
    st.header("AI 컨설턴트의 보고서 초안")
    
    # 필터링 태그를 사용자에게 보여주지 않기 위해 답변에서 제거
    display_response = st.session_state.ai_response
    display_response = re.sub(r'\[FILTER_COUNTRY:\s*.*?\]\s*\[FILTER_YEAR:\s*.*?\]', '', display_response, flags=re.DOTALL | re.MULTILINE).strip()
    
    st.info(display_response)
    
    # 2. 시각적 검증 섹션
    st.markdown("---")
    st.header("2. Tableau를 통한 시각적 검증")
    
    # 필터링 URL 생성 및 임베딩
    tableau_url = get_filtered_tableau_url(
        TABLEAU_BASE_URL, 
        st.session_state.filtered_country,
        st.session_state.filtered_year,
        st.session_state.filtered_region # 세 번째 필터 추가
    )
    
    filter_status = f"""
    **✅ 동적 파라미터 적용 성공:**
    - **국가 ({TABLEAU_FILTER_FIELD_COUNTRY}):** `{st.session_state.filtered_country or '기본값 적용'}` 
    - **연도 ({TABLEAU_FILTER_FIELD_YEAR}):** `{st.session_state.filtered_year or '최신 연도 적용'}`
    - **지역 ({TABLEAU_FILTER_FIELD_REGION}):** `{st.session_state.filtered_region or '기본 지역 적용'}`
    """
    
    if st.session_state.filtered_country and st.session_state.filtered_year and st.session_state.filtered_region:
        st.markdown(filter_status)
        st.code(f"생성된 Tableau URL (파라미터 포함): {tableau_url}", language="url")
    else:
        st.warning(f"⚠️ 단일 국가/년도/지역 추출에 실패했습니다. (AI가 태그 포맷을 어김). 현재 상태: {filter_status}")
        st.info(f"기본 URL (필터링 없음): {TABLEAU_BASE_URL}")

    # Tableau 대시보드를 Streamlit에 HTML iFrame으로 임베딩
    st.components.v1.iframe(tableau_url, height=1300, scrolling=True)
