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
# Initial setup to use 'wide' layout and address visibility issues with Dark theme.
st.set_page_config(layout="wide")

# =================================================================
# [REQUIRED CONFIG] 1. Enter API Key, File, and Tableau Information
# =================================================================

# 1. Reflect user-provided API Key (Updated with the key from your latest message)
# 🚨 API Key is always updated to the latest version.
OPENAI_API_KEY = "Your OPEN AI Key".strip()

# 2. Reflect user-provided Tableau URL (HIV Dashboard URL)
TABLEAU_BASE_URL = "https://public.tableau.com/views/UNICEFHIVTech_AI/UNICEFHIVTechAI?:showVizHome=no&:embed=true"

# 3. Set Tableau Parameter Names
TABLEAU_FILTER_FIELD_COUNTRY = "country" # Tableau Filter Name (lowercase 'country' used for filtering)
TABLEAU_FILTER_FIELD_YEAR = "Year" 
TABLEAU_FILTER_FIELD_REGION = "Unicef Region" 

# Data filename setting
DATA_FILENAME = "unicef_hiv_tech.csv"

# =================================================================
# [DATA LOADING AND OPTIMIZATION]
# =================================================================
@st.cache_data
def load_and_prepare_data(filename):
    """Loads the CSV file and optimizes data for AI analysis."""
    try:
        df = pd.read_csv(filename, encoding='utf-8')
        # 1. Drop NaNs in essential columns
        df_filtered = df.dropna(subset=['country', 'year', 'PLHIV_0_19', 'ART_Coverage_0_14_Pct', 'Annual_New_Infections_0_14']).copy()

        if df_filtered.empty:
             st.warning("⚠️ Data is empty after filtering. AI analysis cannot be run.")
             return pd.DataFrame()

        # 2. Convert Year to integer type
        df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce').astype('Int64')
        
        # 3. Extract only the last 5 years of data 
        latest_year = df_filtered['year'].max()
        start_year = latest_year - 4
        # 🚨 FIX: Perform filtering only if df_filtered is not empty
        df_sample = df_filtered[df_filtered['year'] >= start_year].copy()
        
        # 4. Filter for top 30 countries by max PLHIV count 
        if not df_sample.empty:
            # 🚨 FIX: If the Dataframe is not empty, retrieve the list of top 30 countries.
            top_30_countries = df_sample.groupby('country')['PLHIV_0_19'].max().nlargest(30).index.tolist()
            # Filter df_sample based on the top 30 countries.
            df_sample = df_sample[df_sample['country'].isin(top_30_countries)] 
        
        return df_sample
    except FileNotFoundError:
        st.error(f"🚨 File not found: Please place the {filename} file in your working folder.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"🚨 Error during data loading: {e}")
        return pd.DataFrame()

df_sample = load_and_prepare_data(DATA_FILENAME)

# 🚨 CORE FIX: Enhanced API client initialization error handling and global variable setting
client = None
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    try:
        # Attempting client initialization
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"🚨 OpenAI Client initialization failed. Please check your API key. Error: {e}")
else:
    st.error("🚨 OpenAI API key is invalid or not set. Please update the OPENAI_API_KEY.")


# =================================================================
# [CORE FUNCTION 1] LLM Call and Response Generation
# =================================================================

def generate_ai_response(user_question, df_sample):
    """Calls the LLM to perform data analysis and generate a draft report."""
    # API client not initialized or data is empty, return error immediately
    if not client: return "Analysis failed: API client not initialized. Check your API key."
    if df_sample.empty: return "Analysis failed: The filtered data set is empty."
        
    # Data summary to provide to the LLM (selecting key indicators)
    data_summary = df_sample[[
        'country', 'year', 'unicef_region', 'PLHIV_0_19', 'ART_Coverage_0_14_Pct',
        'Annual_New_Infections_0_14', 'Annual_AIDS_Deaths_0_14', 'MTCT_Rate_Pct'
    ]].to_markdown(index=False)

    # Set default values
    latest_year_in_data = str(df_sample['year'].max())
    default_country_name = 'South Africa'
    
    # 🌟 PROMPT MODIFICATION: ENFORCING ENGLISH OUTPUT 🌟
    system_prompt = f"""
You are an **AI-powered HIV Policy Consultant for UNICEF**.
Answer the user's question and draft a policy report based on the provided CSV data (latest 5 years data for the top 30 countries by PLHIV count).
### [Analysis Data Summary (Latest 5 Years, Top 30 Countries)]
{data_summary}
### [Analysis Guidelines and Data Definitions]
1. **Data-Driven:** Answers must be based strictly on the factual data presented in the table above.
2. **Forced Selection and Defaults:** You **MUST** select a single country and a single year for Tableau dashboard control. If no answer is found, you **MUST** use the defaults: '{default_country_name}' and '{latest_year_in_data}'.
3. **Draft Report:** Focus the report on the selected **single country**, utilizing the PLHIV_0_19, ART_Coverage_0_14_Pct, and Annual_New_Infections_0_14 fields. The report MUST start with the exact text: **Draft Report** followed by a newline, and use clean, standard markdown formatting (titles, lists).
4. **Language ENFORCEMENT:** **You MUST write the entire response, including the analysis, the draft report, and any other text, ONLY in English. Absolutely NO Korean or other languages should be present in the final output, regardless of the user's input language.**

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
**Draft Report**
1. Slide 1. [Title: HIV Response Status in Selected Country]: Summarize the trend of ART Coverage (`ART_Coverage_0_14_Pct`) and New Infections (`Annual_New_Infections_0_14`) for the **single country** selected based on data (`PLHIV_0_19` utilization). Use a markdown list for trends.
2. Slide 2. [Title: Policy Intervention Recommendation]: Based on the country's MTCT Rate (`MTCT_Rate_Pct`) and mortality rate (`Annual_AIDS_Deaths_0_14`) data, propose two immediate policy recommendations for UNICEF to implement. Use a markdown list for recommendations.
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
        return f"Analysis failed due to an API call error. Error: {e}"

# 🌟 [CORE FUNCTION 3] LLM Call to generate Next Step Guide
def generate_next_step_guide(filtered_country, current_analysis_summary):
    """
    Calls the LLM to generate three specific, sequential next-step questions 
    based on the current analysis summary and the selected country.
    """
    # UI Text fully translated to English
    if not client: return "1. Failed to generate next step questions (API Error). \n2. Please check your API key and balance. \n3. Or try again later."
    
    # Preventing empty summary
    current_analysis_summary = current_analysis_summary or "No analysis summary available."
    
    # Summarize if AI response is too long
    if len(current_analysis_summary) > 500:
        current_analysis_summary = current_analysis_summary[:500] + "..."

    system_prompt = f"""
You are an expert analytical guide for UNICEF policy analysts.
Your task is to review the preliminary analysis provided for the country '{filtered_country}' and generate exactly three, specific, sequential follow-up questions to drive deeper, data-informed policy research.
The output MUST be a clean, numbered list using English (1., 2., 3.) without any introductory or concluding remarks.
Ensure the questions are specific to the current context of {filtered_country} (e.g., if ART coverage is low, ask about access barriers).

### [Current Analysis Context]
- **Country Selected:** {filtered_country}
- **Current Analysis Summary:** {current_analysis_summary}

### [Output Requirements]
1.  **Focus:** Questions must be specific to **{filtered_country}**.
2.  **Sequence:** Questions should naturally progress (e.g., Diagnosis -> Root Cause -> Intervention).
3.  **Format:** A numbered list (1., 2., 3.) only, in **English**. Do not use markdown list syntax (* or -), use explicit numbers followed by a period (1., 2., 3.).
"""
    user_prompt = "Generate the three specific, next-step policy analysis questions for the selected country."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # UI Text fully translated to English
        return f"1. Failed to generate next step questions (API Error). \n2. Error: {e}"

# =================================================================
# [CORE FUNCTION 2] Extract Structured Single Country, Year, Region and Generate URL
# =================================================================

def extract_structured_filter_value(ai_response, tag):
    """Extracts the value from the structured filter tag ([TAG: Value])."""
    # 🚨 FIX: Modified regex to handle newlines and spaces flexibly (using re.DOTALL flag)
    match = re.search(r'\[' + re.escape(tag) + r':\s*(.*?)\]', ai_response, flags=re.DOTALL | re.IGNORECASE)
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
    # 🚨 FIX: Prevent calling max() if data is empty
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
    """Generates the Tableau URL with the country filter applied."""
    filter_key_raw = f"p.{TABLEAU_FILTER_FIELD_COUNTRY}" 
    
    if not country_value:
        return base_url 
    
    encoded_key = quote_plus(filter_key_raw) 
    encoded_value = quote_plus(country_value)
    
    filter_query_string = f"&{encoded_key}={encoded_value}"
    
    final_url = base_url + filter_query_string

    return final_url

# =================================================================
# [Session State Management]
# =================================================================

default_question = """UNICEF's core priorities are 'scaling up treatment for children with HIV and minimizing new infections.'\nUsing data analysis:\n1. Identify countries with a **large PLHIV (0-19) scale but an ART Coverage (ART_Coverage_0_14_Pct) below 60%**.\n2. From this group, select the **single country** that appears to have the **slowest decrease or an increase in new infections (Annual_New_Infections_0_14)** over the last 5 years, and infer the reasons for the response failure (e.g., MTCT_Rate_Pct).\n3. Draft a policy report focusing on the **single country** selected."""

# Initialize and manage session state
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'filtered_country' not in st.session_state:
    # UI Text fully translated to English
    st.session_state.filtered_country = "Pre-Analysis"
if 'filtered_year' not in st.session_state:
    st.session_state.filtered_year = ""
if 'filtered_region' not in st.session_state:
    st.session_state.filtered_region = ""
if 'next_step_guide' not in st.session_state:
    # UI Text fully translated to English
    st.session_state.next_step_guide = """
    **Logical Flow for this Dashboard Session:**
    
    1. Identify Key Country (Completed):
       - Q: Which country is most vulnerable?
    
    2. Deep Dive Diagnosis (Next Question):
       - Q: What are the geographical causes of the treatment gap (ART Coverage) in the **selected country**?
    
    3. Policy Proposal (Next Question):
       - Q: What specific policy and budget allocation proposals should be made for regions with high MTCT in the **selected country**?
    
    💡 Tip: Structuring your questions sequentially leads the AI to produce more accurate and in-depth reports.
    """
if 'user_prompt_text_area' not in st.session_state:
    st.session_state.user_prompt_text_area = default_question


# =================================================================
# 🎨 [DESIGNED UI STRUCTURE] Streamlit UI Construction
# =================================================================

# --- 1. Custom CSS and Header ---
# 🚨 CSS comments translated to English
st.markdown("""
<style>
    /* 1. Force bright global font color for visibility in Dark Mode without dragging */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] * { 
        color: #E5E7EB !important; 
    }
    
    /* 2. Custom Tailwind Colors for Streamlit Components */
    .primary-blue { color: #1080FF; }
    .secondary-gray { background-color: #374151; } /* Adjusted background color for Dark Mode */
    .accent-green { color: #1ABF79; }
    
    /* 3. Info Box Style (AI Analysis Result) */
    .info-box-custom {
        background-color: #1F2937; /* Dark background for Dark Mode */
        border-left: 5px solid #1080FF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #E5E7EB; /* Light text color */
    }
    
    /* 4. Text Area (Input Area) */
    .stTextArea > label { font-size: 0.875rem; font-weight: 500; color: #9CA3AF; }
    .stTextArea > div > textarea { 
        border-radius: 0.5rem; 
        border-color: #4B5563; /* Darker border color */
        padding: 0.75rem;
        background-color: #1F2937; /* Darker background color */
        color: #E5E7EB; /* Light input text color */
    }

    /* 5. Custom Header and Subheader styles (Ensures title visibility) */
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        color: #F9FAFB; 
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.125rem;
        color: #D1D5DB; 
    }
    .text-xl { color: #F9FAFB !important; } /* Section Title (H2) visibility */
    .text-lg { color: #F9FAFB !important; } /* Section Title (H3) visibility */
    
    /* Streamlit h2, h3 default color override (fixes issue where titles were invisible) */
    h2, h3 { color: #F9FAFB !important; }


    /* 6. Next Analysis Guide text color and container adjustment */
    .guide-box p, .guide-box ol li { color: #E5E7EB !important; }
    
    /* 7. Button Style */
    .stButton>button {
        background-color: #1080FF; 
        color: white; 
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        transition: all 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1C6EDD;
    }
</style>
""", unsafe_allow_html=True)

# Application Header (UI Text fully translated to English)
st.markdown(f"""
    <header class="mb-6">
        <h1 class="main-header">💡 AI Driven HIV Response Dashboard Demo (UNICEF Data)</h1>
        <h3 class="sub-header">
            🗣️ **Conversational AI Analysis** automatically sets dashboard filters and generates policy reports.
        </h3>
        <div class="h-px bg-gray-700 mt-4"></div>
    </header>
""", unsafe_allow_html=True)


# --- 2. Top Section: Chat, Analysis, and Guide (3:1 Column Layout) ---
col_chat, col_guide = st.columns([3, 1])

# 2.1. Left Column: Question Input and Analysis Results
with col_chat:
    # UI Text fully translated to English
    st.markdown('<h2 class="text-xl font-bold mb-4">1. AI-Powered Question and Report Generation</h2>', unsafe_allow_html=True)
    
    # Prompt Input Area (UI Text fully translated to English)
    user_question = st.text_area(
        "📝 Ask the AI HIV Consultant a question:",
        value=st.session_state.user_prompt_text_area,
        key='user_prompt_text_area', # Mandatory: maintains input value
        height=180
    )

    # Request Button (UI Text fully translated to English)
    if st.button("🚀 Request AI Analysis and Filter Dashboard", type="primary", use_container_width=True):
        if df_sample.empty:
            st.error("🚨 The data file lacks sufficient valid data for analysis. Please check the file contents.")
            st.stop()
        
        # 1. Call LLM for analysis
        # UI Text fully translated to English
        with st.spinner("AI is analyzing UNICEF data and updating Tableau parameters..."):
            ai_result = generate_ai_response(user_question, df_sample)
            st.session_state.ai_response = ai_result
            
            # 2. Extract filters
            st.session_state.filtered_country = extract_single_country(ai_result, df_sample)
            st.session_state.filtered_year = extract_single_year(ai_result, df_sample)
            st.session_state.filtered_region = extract_region_from_country(st.session_state.filtered_country, df_sample)
            
            # --- 3. [CRITICAL FIX] Clean analysis text using robust split logic ---
            report_start_marker = '**Draft Report**'
            
            # 3a. Remove tag (before generating guide/summary)
            clean_response_text = re.sub(r'\[FILTER_COUNTRY:\s*.*?\]\s*\[FILTER_YEAR:\s*.*?\]', '', st.session_state.ai_response, flags=re.DOTALL | re.MULTILINE).strip()
            
            # 3b. Extract analysis answer part only (content before 'Draft Report')
            analysis_parts = clean_response_text.split(report_start_marker, 1)
            
            if len(analysis_parts) > 1:
                # First part = Analysis Answer
                analysis_answer_only = analysis_parts[0].strip()
            else:
                # Fallback: use the whole response if marker is missing
                # UI Text fully translated to English
                analysis_answer_only = "Failed to extract analysis answer from the response. Using original response."

            # 4. Call LLM for dynamic next step guide 
            st.session_state.next_step_guide = generate_next_step_guide(st.session_state.filtered_country, analysis_answer_only)

        if "API call error" not in st.session_state.ai_response and "Analysis failed" not in st.session_state.ai_response:
            # UI Text fully translated to English
            st.success("Analysis complete! Check the AI's interpretation and the automatically parameterized dashboard.")
        else:
            # UI Text fully translated to English
            st.error("🚨 Analysis failed! Please check your API key, balance, or data file.")

    # AI Analysis Summary 
    if st.session_state.ai_response:
        display_response_text = st.session_state.ai_response
        # Remove tags
        display_response_text = re.sub(r'\[FILTER_COUNTRY:\s*.*?\]\s*\[FILTER_YEAR:\s*.*?\]', '', display_response_text, flags=re.DOTALL | re.MULTILINE).strip()
        
        # Extract analysis answer summary only (content before 'Draft Report')
        report_start_marker = '**Draft Report**'
        analysis_parts = display_response_text.split(report_start_marker, 1)
        
        if len(analysis_parts) > 1:
            analysis_answer_only_display = analysis_parts[0].strip()
        else:
            analysis_answer_only_display = display_response_text.strip() # Fallback
        
        # UI Text fully translated to English
        st.markdown('<h3 class="text-lg font-semibold mt-6 mb-2">✅ AI Analysis Key Findings</h3>', unsafe_allow_html=True)
        
        # Apply styled info-box (UI Text fully translated to English)
        st.markdown(
            f"""
            <div class="info-box-custom">
                <p class="font-bold primary-blue mb-1">Selected Country: {st.session_state.filtered_country}</p>
                <p class="text-sm" style="white-space: pre-wrap; color: #E5E7EB;">
                    <span class="font-medium">AI Analysis Answer Summary:</span> {analysis_answer_only_display}
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# 2.2. Right Column: Insight and Guide 
with col_guide:
    # UI Text fully translated to English
    st.markdown('<h3 class="text-lg font-semibold mb-4">📚 Next Step Analysis Guide</h3>', unsafe_allow_html=True)
    # Apply styled guide box (UI Text fully translated to English)
    st.markdown(f"""
    <div class="secondary-gray guide-box p-4 rounded-xl shadow-inner" style="min-height: 240px; white-space: pre-wrap;">
        <p class="font-bold mb-3">
            Deep dive suggestions for (<span class="accent-green">{st.session_state.filtered_country or 'Pre-Analysis'}</span>):
        </p>
        <p class="text-sm" style="color: #E5E7EB;">
            {st.session_state.next_step_guide}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="h-px bg-gray-700 mt-6 mb-6"></div>', unsafe_allow_html=True)


# --- 3. Bottom Section: Tabs for Dashboard and Report ---
# UI Text fully translated to English
tab1, tab2 = st.tabs(["📊 Tableau Visual Verification (Filter Applied)", "📄 AI Policy Draft Report"])

# 3.1. Tab 1: Tableau Dashboard
with tab1:
    # UI Text fully translated to English
    st.markdown('<h2 class="text-xl font-bold mb-4">2. Tableau Dashboard Visual Verification</h2>', unsafe_allow_html=True)
    
    tableau_url = get_filtered_tableau_url(
        TABLEAU_BASE_URL,
        st.session_state.filtered_country,
        st.session_state.filtered_year,
        st.session_state.filtered_region
    )
    
    filter_status = f"""
    - Country (p.{TABLEAU_FILTER_FIELD_COUNTRY}): <strong class="primary-blue">{st.session_state.filtered_country or 'Default Applied'}</strong> (Automatically Applied)
    - Year ({TABLEAU_FILTER_FIELD_YEAR}): <strong>{st.session_state.filtered_year or 'Latest Year Applied'}</strong> (Not applied to Tableau filter)
    """
    
    # Apply styled status box (UI Text fully translated to English)
    st.markdown(f"""
    <div class="mb-4 text-sm bg-gray-800 p-3 rounded-lg border border-gray-700" style="color: #E5E7EB;">
        <p class="font-bold accent-green mb-1">✅ Dynamic Parameter Application Status (Only Country Filter Applied):</p>
        <p class="text-gray-400">{filter_status}</p>
    </div>
    """, unsafe_allow_html=True)

    # Embed Tableau dashboard
    st.components.v1.iframe(tableau_url, height=900, scrolling=True)

# 3.2. Tab 2: Draft Report 
with tab2:
    if st.session_state.ai_response and "Analysis failed" not in st.session_state.ai_response:
        # UI Text fully translated to English
        st.markdown(f'<h2 class="text-xl font-bold mb-4">AI Consultant\'s Draft Report for {st.session_state.filtered_country}</h2>', unsafe_allow_html=True)
        
        # Extract only the report part: everything after "**Draft Report**" and before the final tag
        report_start_marker = '**Draft Report**'
        
        # Split response into parts
        analysis_parts = st.session_state.ai_response.split(report_start_marker, 1)

        if len(analysis_parts) > 1:
            # Second part = Report Content + Filter Tag
            draft_report_content = analysis_parts[1].strip()
            
            # Remove final filter tag
            draft_report_content = re.sub(r'\[FILTER_COUNTRY:\s*.*?\]\s*\[FILTER_YEAR:\s*.*?\]', '', draft_report_content, flags=re.DOTALL | re.MULTILINE).strip()
            
            # Use st.markdown for proper rendering of the markdown report
            st.markdown(draft_report_content)
        else:
            # Provide debugging info if extraction fails (UI Text fully translated to English)
            st.warning(f"⚠️ Failed to extract the draft report section from the AI response. Please ensure the AI used the '**Draft Report**' marker.")
            st.info(f"AI Response Original (for debug): \n{st.session_state.ai_response}")
            
    else:
        # UI Text fully translated to English
        st.info("Please click the 'Request AI Analysis' button first to generate the analysis results.")
