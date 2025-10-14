import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import os
import numpy as np
# Import quote_plus for URL encoding.
from urllib.parse import quote_plus

# =================================================================
# 🚨 [TOP-LEVEL CONFIG] Page configuration must be the very first Streamlit command.
# =================================================================
st.set_page_config(layout="wide")

# =================================================================
# [REQUIRED CONFIG] 1. Enter API Key, File, and Tableau Information
# =================================================================

# 1. Reflect user-provided API Key (updated with new key)
# 🚨 This is the new key.
OPENAI_API_KEY = "yourOpenAI API Key".strip()

# 2. Reflect user-provided Tableau URL (HIV Dashboard URL)
TABLEAU_BASE_URL = "https://public.tableau.com/views/UNICEFHIVTech_AI/UNICEFHIVTechAI?:showVizHome=no&:embed=true"

# 3. 🚨 Set Tableau Parameter Names (using all lowercase country field name)
TABLEAU_FILTER_FIELD_COUNTRY = "country" # 🚨🚨🚨 FINAL CHANGE: Tableau Filter Name set to 'country' (all lowercase)
TABLEAU_FILTER_FIELD_YEAR = "Year" # Tableau Filter Name: Year (KEPT FOR LLM OUTPUT/REFERENCE)
TABLEAU_FILTER_FIELD_REGION = "Unicef Region" # Tableau Filter Name: Unicef Region (KEPT FOR LLM OUTPUT/REFERENCE)

# Data filename setting
DATA_FILENAME = "unicef_hiv_tech.csv"

# =================================================================
# [DATA LOADING AND OPTIMIZATION]
# =================================================================
@st.cache_data
def load_and_prepare_data(filename):
    """Loads the CSV file and optimizes data for AI analysis."""
    try:
        # Load file (can change to 'latin-1' if encoding issue occurs)
        df = pd.read_csv(filename, encoding='utf-8')
        # 1. Drop NaNs in essential columns: Only rows with core indicators (PLHIV, ART, New Infections) are used for analysis
        df_filtered = df.dropna(subset=['country', 'year', 'PLHIV_0_19', 'ART_Coverage_0_14_Pct', 'Annual_New_Infections_0_14']).copy()

        # 2. Convert Year to integer type
        df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce').astype('Int64')
        # 3. Extract only the last 5 years of data for the LLM prompt
        if not df_filtered.empty:
            latest_year = df_filtered['year'].max()
            start_year = latest_year - 4
            df_sample = df_filtered[df_filtered['year'] >= start_year].copy()
            # 4. Filter for top 30 countries by max PLHIV count to increase LLM analysis efficiency
            top_30_countries = df_sample.groupby('country')['PLHIV_0_19'].max().nlargest(30).index.tolist()
            df_sample = df_sample[df_sample['country'].isin(top_30_countries)]
        else:
            df_sample = df_filtered
        return df_sample
    except FileNotFoundError:
        st.error(f"🚨 File not found: Please place the {filename} file in your VS Code working folder.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Error during data loading: {e}")
        st.stop()

df_sample = load_and_prepare_data(DATA_FILENAME)

# Initialize OpenAI client and validate key
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.error("🚨 OpenAI API key is invalid or not set.")
    client = None
    # st.stop() # 주석 처리: 키가 없어도 UI는 볼 수 있도록

# =================================================================
# [CORE FUNCTION 1 & 3] LLM Call and Response Generation
# =================================================================

def generate_ai_response(user_question, df_sample):
    """Calls the LLM to perform data analysis and generate a draft report."""
    if not client: return "Analysis cannot be run due to API client error. Please check your key."
    # Data summary to provide to the LLM (selecting key indicators)
    data_summary = df_sample[[
        'country', 'year', 'unicef_region', 'PLHIV_0_19', 'ART_Coverage_0_14_Pct',
        'Annual_New_Infections_0_14', 'Annual_AIDS_Deaths_0_14', 'MTCT_Rate_Pct'
    ]].to_markdown(index=False)

    # Set default values
    latest_year_in_data = str(df_sample['year'].max() if not df_sample.empty else 2024)
    default_country_name = 'South Africa'
    # 🌟 PROMPT MODIFICATION: Adjusted to fit UNICEF HIV data and question 🌟
    system_prompt = f"""
You are an **AI-powered HIV Policy Consultant for UNICEF**.
Answer the user's question and draft a policy report based on the provided CSV data (latest 5 years data for the top 30 countries by PLHIV count).
### [Analysis Data Summary (Latest 5 Years, Top 30 Countries)]
{data_summary}
### [Analysis Guidelines and Data Definitions]
1. **Data-Driven:** Answers must be based strictly on the factual data presented in the table above.
2. **Forced Selection and Defaults:** You **MUST** select a single country and a single year for Tableau dashboard control. If no answer is found, you **MUST** use the defaults: '{default_country_name}' and '{latest_year_in_data}'.
3. **Draft Report:** Focus the report on the selected **single country**, utilizing the PLHIV_0_19, ART_Coverage_0_14_Pct, and Annual_New_Infections_0_14 fields.

### 🚨🚨🚨 [ABSOLUTE ENFORCEMENT OF FINAL OUTPUT TAG FORMAT] 🚨🚨🚨
1. After all response content (including the analysis report) is complete, use the tag **only on the very last line, standalone**.
2. **Exact Format:** [FILTER_COUNTRY: [Country English Name]][FILTER_YEAR: [Year Number]]
3. **Country Name:** Use only the **English country name** present in the dataset.
4. **Example (DO NOT deviate from this format):** [FILTER_COUNTRY: {default_country_name}][FILTER_YEAR: {latest_year_in_data}]
5. **Warning:** Absolutely no other text, spaces, line breaks, or explanations should be added to the last line other than the tag string itself.
"""
    # Integrate user question and report draft request
    prompt_with_report = f"""
[User Question]: {user_question}
First, provide the analysis answer, and then write a draft report following the Powerpoint slide format below.
[Draft Report] (Focus on the selected single country)
1. Slide 1. [Title: HIV Response Status in Selected Country]: Summarize the trend of ART Coverage (`ART_Coverage_0_14_Pct`) and New Infections (`Annual_New_Infections_0_14`) for the **single country** selected based on data (`PLHIV_0_19` utilization).
2. Slide 2. [Title: Policy Intervention Recommendation]: Based on the country's MTCT Rate (`MTCT_Rate_Pct`) and mortality rate (`Annual_AIDS_Deaths_0_14`) data, propose two immediate policy recommendations for UNICEF to implement.
**✅ After providing all answers and the report, insert the filtering tag on the very last line as per system instructions.**
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
        # Return the API error message directly to the user
        return f"Analysis failed due to an error during the API call. Error: {e}"

# =================================================================
# [CORE FUNCTION 2] Extract Structured Single Country, Year, Region and Generate URL
# =================================================================

def extract_structured_filter_value(ai_response, tag):
    """Extracts the value from the structured filter tag ([TAG: Value])."""
    match = re.search(r'\[' + re.escape(tag) + r':\s*(.*?)\]', ai_response)
    if match:
        return match.group(1).strip()
    return None

def extract_single_country(ai_response, df_sample):
    """Extracts the structured country name from the AI response text."""
    country = extract_structured_filter_value(ai_response, "FILTER_COUNTRY")
    if country and country in df_sample['country'].unique().tolist():
        return country
    return 'South Africa' # Default value


def extract_single_year(ai_response, df_sample):
    """Extracts the structured year number from the AI response text."""
    year_str = extract_structured_filter_value(ai_response, "FILTER_YEAR")
    latest_year = str(df_sample["year"].max() if not df_sample.empty else 2024)

    if year_str and year_str.isdigit():
        year = int(year_str)
        if year in df_sample["year"].unique().tolist():
            return year_str
    return latest_year # Default: latest year


def extract_region_from_country(country_value, df_sample):
    """Looks up the UNICEF Region from the DataFrame based on the selected country."""
    if country_value and country_value in df_sample['country'].unique().tolist() and not df_sample.empty:
        # Find the Region based on the latest data for that country.
        latest_data = df_sample[df_sample['country'] == country_value].sort_values(by='year', ascending=False).iloc[0]
        return str(latest_data['unicef_region'])
    return 'Eastern and Southern Africa' # Default value


def get_filtered_tableau_url(base_url, country_value, year_value, region_value):
    """
    [최종 수정됨] Tableau URL의 기존 뷰 매개변수(?:...) 형식을 유지하고,
    그 뒤에 '&p.country=Value' 형식으로 Country 필터만 추가하여 생성합니다.
    """
    # 1. 필터 파라미터 (p.country) 키와 값 준비 (필터 이름이 소문자 'country'로 변경됨)
    filter_key_raw = f"p.{TABLEAU_FILTER_FIELD_COUNTRY}" # 예: "p.country"
    
    if not country_value:
        # 국가 값이 없으면 필터링하지 않고 기본 URL 반환
        return base_url 
    
    # 파라미터 이름(키)과 값(value)을 모두 URL 인코딩합니다.
    # 예: p.country -> p.country, United States -> United+States
    encoded_key = quote_plus(filter_key_raw) 
    encoded_value = quote_plus(country_value)
    
    # 2. 필터 쿼리 문자열 생성 (&p.country=angola)
    # 반드시 '&'로 시작해야 기존 뷰 파라미터 뒤에 붙일 수 있습니다.
    filter_query_string = f"&{encoded_key}={encoded_value}"
    
    # 3. 기존 base_url에 필터 쿼리 문자열을 추가합니다.
    # base_url이 이미 ?:showVizHome=no&:embed=true 형식으로 끝납니다.
    final_url = base_url + filter_query_string

    return final_url

# =================================================================
# Streamlit UI Construction
# =================================================================

st.title("💡 AI Driven HIV Response Dashboard Demo (UNICEF Data)")
st.markdown("### ⚠️ **Note:** The Tableau dashboard is controlled by **standard query string parameters**, but currently **only the Country filter is applied**.")
st.markdown("---")

# 1. Conversational Question Section
st.header("1. AI-Powered Question and Report Generation")

# User request: Use the provided question as default
default_question = """
UNICEF's core priorities are 'scaling up treatment for children with HIV and minimizing new infections.'
Using data analysis:
1. Identify countries with a **large PLHIV (0-19) scale but an ART Coverage (ART_Coverage_0_14_Pct) below 60%**.
2. From this group, select the **single country** that appears to have the **slowest decrease or an increase in new infections (Annual_New_Infections_0_14)** over the last 5 years, and infer the reasons for the response failure (e.g., MTCT_Rate_Pct).
3. Draft a policy report focusing on the **single country** selected.
"""

user_question = st.text_area(
    "Ask the AI HIV Consultant a question:",
    default_question,
    height=200
)

# Initialize and manage session state
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'filtered_country' not in st.session_state:
    st.session_state.filtered_country = ""
if 'filtered_year' not in st.session_state:
    st.session_state.filtered_year = ""
if 'filtered_region' not in st.session_state:
    st.session_state.filtered_region = ""


if st.button("🚀 Request AI Analysis and Filter Dashboard", type="primary"):
    if df_sample.empty:
        st.error("🚨 The data file lacks sufficient valid data for analysis. Please check the file contents.")
        st.stop()
    with st.spinner("AI is analyzing UNICEF data and updating Tableau parameters..."):
        # 1. Call LLM and generate response
        ai_result = generate_ai_response(user_question, df_sample)
        st.session_state.ai_response = ai_result
        # 2. Extract single country from response
        selected_country = extract_single_country(ai_result, df_sample)
        st.session_state.filtered_country = selected_country
        # 3. Extract single year from response
        st.session_state.filtered_year = extract_single_year(ai_result, df_sample)
        # 4. Look up Region based on the extracted country (using the DataFrame)
        st.session_state.filtered_region = extract_region_from_country(selected_country, df_sample)


    if "API call failed" not in st.session_state.ai_response:
        st.success("Analysis complete! Check the AI's interpretation and the automatically parameterized dashboard.")
    else:
        st.error("🚨 Analysis failed! Please check your API key and balance.")


if st.session_state.ai_response:
    # Output LLM response
    st.markdown("---")
    st.header("AI Consultant's Draft Report")
    # Remove filtering tag from the response before displaying to the user
    display_response = st.session_state.ai_response
    display_response = re.sub(r'\[FILTER_COUNTRY:\s*.*?\]\s*\[FILTER_YEAR:\s*.*?\]', '', display_response, flags=re.DOTALL | re.MULTILINE).strip()
    st.info(display_response)
    # 2. Visual Verification Section
    st.markdown("---")
    st.header("2. Visual Verification via Tableau")
    # Generate filtered URL
    tableau_url = get_filtered_tableau_url(
        TABLEAU_BASE_URL,
        st.session_state.filtered_country,
        st.session_state.filtered_year,
        st.session_state.filtered_region
    )
    # 🚨 Display the final, corrected filter key (p.country)
    filter_status = f"""
**✅ Dynamic Parameter Application Status (Only Country Applied):**
- **Country (p.{TABLEAU_FILTER_FIELD_COUNTRY}):** `{st.session_state.filtered_country or 'Default Applied'}`
- **Year ({TABLEAU_FILTER_FIELD_YEAR}):** `{st.session_state.filtered_year or 'Latest Year Applied'}` (Not used for Tableau filtering)
- **Region ({TABLEAU_FILTER_FIELD_REGION}):** `{st.session_state.filtered_region or 'Default Region Applied'}` (Not used for Tableau filtering)
"""
    if st.session_state.filtered_country:
        st.markdown(filter_status)
        st.code(f"Generated Tableau URL (with correct p.country parameter format): {tableau_url}", language="url")
    else:
        st.warning(f"⚠️ Failed to extract single country. (AI violated tag format). Current Status: {filter_status}")
        st.info(f"Base URL (No filtering): {TABLEAU_BASE_URL}")

    # Embed Tableau dashboard using an HTML iFrame
    st.components.v1.iframe(tableau_url, height=1300, scrolling=True)
