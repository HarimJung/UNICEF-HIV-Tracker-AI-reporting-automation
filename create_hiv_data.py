import pandas as pd
import numpy as np
import os

# 🚨 파일 경로 설정: 'Data' 폴더 안에 파일이 있어야 합니다.
FILE_PATH = 'Data/' 
ORPHANS_FILE = os.path.join(FILE_PATH, "HIV_Orphans_2025.csv")
EPI_FILE = os.path.join(FILE_PATH, "HIV_Epidemiology_Children_Adolescents_2025.csv")
ART_FILE = os.path.join(FILE_PATH, "HIV_Paediatric_ART_Coverage_2025.csv")
FINAL_FILENAME = 'unicef_hiv_tech.csv'


# ----------------------------------------------------
# 1. 공통 데이터 정리 함수: 콤마, '<' 기호 처리 로직 포함
# ----------------------------------------------------
def clean_and_pivot_data(df, indicator_name, new_col_name, age_sex_filter=None, pivot_by_region=False):
    """지정된 지표를 필터링하고 와이드 형식으로 피벗하는 함수"""
    
    # 1. Country-level 데이터만 필터링 
    if df is None or 'Type' not in df.columns:
        return pd.DataFrame()
        
    df_filtered = df[df['Type'] == 'Country'].copy()
    
    # 2. 지정된 Indicator만 선택
    df_filtered = df_filtered[df_filtered['Indicator'] == indicator_name].copy()
    
    # 3. 추가적인 Age/Sex 필터 적용
    if age_sex_filter:
        for col, value in age_sex_filter.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == value].copy()
            
    # 4. Value 열 정리 (콤마, 따옴표, '<' 기호 처리 로직 최종 보강)
    def clean_value(val):
        if isinstance(val, str):
            # 1단계: 따옴표와 콤마를 모두 제거
            cleaned_val = val.replace('"', '').replace(',', '').strip()
            
            # 2단계: '<' 기호 처리 (최솟값 대체)
            if cleaned_val.startswith('<'):
                if cleaned_val == '<1':
                    return 0.5  # 규모 지표
                elif cleaned_val == '<0.01':
                    return 0.005 # 비율/율 지표
                # 그 외의 '<'는 기호만 제거 후 변환 시도
                cleaned_val = cleaned_val[1:].strip() 
            
            # 3단계: 숫자로 변환 시도
            try:
                if not cleaned_val or cleaned_val == '-':
                    return np.nan
                return pd.to_numeric(cleaned_val)
            except ValueError:
                return np.nan
        # 숫자인 경우 그대로 반환 (NaN 포함)
        return val

    # 'Value' 컬럼 존재 여부 확인 후 적용
    if 'Value' not in df_filtered.columns:
        return pd.DataFrame()
        
    df_filtered['Value'] = df_filtered['Value'].apply(clean_value)
    
    # 5. 최종 데이터프레임 구성
    if pivot_by_region:
        cols = ['ISO3', 'Country/Region', 'UNICEF Region', 'Year', 'Value']
        df_result = df_filtered[cols].rename(
            columns={'Country/Region': 'country', 'Year': 'year', 'Value': new_col_name, 'UNICEF Region': 'unicef_region'}
        )
        subset_cols = ['ISO3', 'country', 'year', 'unicef_region']
    else:
        cols = ['ISO3', 'Country/Region', 'Year', 'Value']
        df_result = df_filtered[cols].rename(
            columns={'Country/Region': 'country', 'Year': 'year', 'Value': new_col_name}
        )
        subset_cols = ['ISO3', 'country', 'year']
    
    # 중복 제거
    df_result = df_result.drop_duplicates(subset=subset_cols, keep='first')
    
    return df_result[subset_cols + [new_col_name]]


# ----------------------------------------------------
# 2. 데이터 파일 로드 및 정리 (14개 필드 추출)
# ----------------------------------------------------
all_dfs = []

# (1) Orphans Data
try:
    # 🚨 파일 구조상 헤더는 0번째 줄 (두 번째 줄)
    df_orphans = pd.read_csv(ORPHANS_FILE, header=0)
    
    # Field 4, 5: Orphans
    indicator_aids = 'Estimated number of children (aged 0-17) who have lost one or both parents due to AIDS'
    df_aids_orphans = clean_and_pivot_data(df_orphans, indicator_aids, 'AIDS_Orphans_0_17', pivot_by_region=True)
    all_dfs.append(df_aids_orphans)

    indicator_all = 'Estimated number of children (aged 0-17) who have lost one or both parents due all causes'
    df_all_orphans = clean_and_pivot_data(df_orphans, indicator_all, 'Orphans_All_Causes_0_17')
    all_dfs.append(df_all_orphans)
except Exception as e:
    print(f"❌ Orphans Data 로드/처리 오류: {ORPHANS_FILE} 파일 로드 실패. 에러: {e}")

# (2) Epidemiology Data
try:
    df_epi = pd.read_csv(EPI_FILE, header=0)
    
    # Field 6, 7, 8, 9, 10, 11, 12: Epidemiology
    indicator_plhiv = 'Estimated number of people living with HIV'
    df_plhiv = clean_and_pivot_data(df_epi, indicator_plhiv, 'PLHIV_0_19', age_sex_filter={'Age': 'Age 0-19', 'Sex': 'Both'})
    all_dfs.append(df_plhiv)

    indicator_rate = 'Estimated number of people living with HIV (per 100,000 population)'
    df_plhiv_rate = clean_and_pivot_data(df_epi, indicator_rate, 'PLHIV_Rate_per_100k_0_19', age_sex_filter={'Age': 'Age 0-19', 'Sex': 'Both'})
    all_dfs.append(df_plhiv_rate)

    indicator_new_num = 'Estimated number of annual new HIV infections'
    df_new_num = clean_and_pivot_data(df_epi, indicator_new_num, 'Annual_New_Infections_0_14', age_sex_filter={'Age': 'Age 0-14', 'Sex': 'Both'})
    all_dfs.append(df_new_num)

    indicator_inc_rate = 'Estimated incidence rate (new HIV infection per 1,000 uninfected population)'
    df_inc_rate = clean_and_pivot_data(df_epi, indicator_inc_rate, 'Incidence_Rate_per_1k_0_14', age_sex_filter={'Age': 'Age 0-14', 'Sex': 'Both'})
    all_dfs.append(df_inc_rate)

    indicator_death_num = 'Estimated number of annual AIDS-related deaths'
    df_death_num = clean_and_pivot_data(df_epi, indicator_death_num, 'Annual_AIDS_Deaths_0_14', age_sex_filter={'Age': 'Age 0-14', 'Sex': 'Both'})
    all_dfs.append(df_death_num)
    
    indicator_death_rate = 'Estimated rate of annual AIDS-related deaths (per 100,000 population)'
    df_death_rate = clean_and_pivot_data(df_epi, indicator_death_rate, 'Death_Rate_per_100k_0_14', age_sex_filter={'Age': 'Age 0-14', 'Sex': 'Both'})
    all_dfs.append(df_death_rate)

    indicator_mtct = 'Estimated mother-to-child transmission rate (%)'
    df_mtct = clean_and_pivot_data(df_epi, indicator_mtct, 'MTCT_Rate_Pct', age_sex_filter={'Age': 'Age 0-4', 'Sex': 'Both'})
    all_dfs.append(df_mtct)
except Exception as e:
    print(f"❌ Epidemiology Data 로드/처리 오류: {EPI_FILE} 파일 로드 실패. 에러: {e}")


# (3) ART Coverage Data
try:
    df_art = pd.read_csv(ART_FILE, header=0)
    
    # Field 13, 14: ART
    indicator_art_cov = 'Per cent of children living with HIV receiving ART'
    df_art_cov = clean_and_pivot_data(df_art, indicator_art_cov, 'ART_Coverage_0_14_Pct')
    all_dfs.append(df_art_cov)

    indicator_art_num = 'Reported number of children receiving ART'
    df_art_num = clean_and_pivot_data(df_art, indicator_art_num, 'Reported_ART_Number')
    all_dfs.append(df_art_num)
except Exception as e:
    print(f"❌ ART Coverage Data 로드/처리 오류: {ART_FILE} 파일 로드 실패. 에러: {e}")


# ----------------------------------------------------
# 3. 데이터 통합 및 저장 (NaN 문자열 변환 제거)
# ----------------------------------------------------

# 비어있는 데이터프레임 제거
all_dfs = [df for df in all_dfs if not df.empty]

if not all_dfs:
    print("🚨 치명적 오류: 모든 데이터 파일 로드 및 정제에 실패했습니다. Data 폴더 내 파일명과 경로를 확인하세요.")
    merged_df = pd.DataFrame(columns=['ISO3', 'country', 'year'])
else:
    # ISO3, country, year를 기준으로 모든 데이터프레임을 병합
    merged_df = all_dfs[0].copy()
    for i in range(1, len(all_dfs)):
        merge_cols = ['ISO3', 'country', 'year']
        if 'unicef_region' in all_dfs[i].columns and 'unicef_region' not in merged_df.columns:
            merge_cols.append('unicef_region')
        
        merged_df = merged_df.merge(all_dfs[i], on=[col for col in merge_cols if col in merged_df.columns and col in all_dfs[i].columns], how='outer')


    # 정렬 및 결측치 NaN 처리: np.nan 값을 그대로 유지합니다. (문자열 'NaN'으로 변환하지 않음)
    merged_df = merged_df.sort_values(by=['country', 'year']).reset_index(drop=True)


# AI 분석용 최종 파일 저장
merged_df.to_csv(FINAL_FILENAME, index=False)

print(f"\n===================================================================")
print(f"✅ 데이터 정제 및 통합 완료! '{FINAL_FILENAME}' 파일이 생성되었습니다.")
print(f"총 {len(merged_df.columns)}개의 필드(컬럼)로 구성되었습니다.")
print(f"===================================================================")
print("\n--- [생성된 데이터 구조 요약 (상위 5개 행)] ---")
print(merged_df.head())