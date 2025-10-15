💡 UNICEF Pediatric HIV Intelligence: AI-Automated Strategic Reporting

## Project Vision: The Zero-Draft Policy Generator

This framework is an advanced, open-source solution dedicated to accelerating high-stakes decision-making in global health. It specifically targets the challenge of translating complex, multi-source **Pediatric HIV indicator data** into **executive-ready policy documents** without manual drafting.

By leveraging an integrated Python stack (ETL, Streamlit, LLM), this system creates a **closed-loop intelligence cycle**. It acts as an **Analyst Augmentation** tool, moving the workflow from data analysis to **final report delivery** in minutes.

-----

## 🚀 Core Value Proposition: The AI Automation Cycle

The system's innovation lies in its ability to automatically generate the four critical outputs required for strategic reporting: **Analysis, Guidance, Insight, and Narrative.**

  * **Automated Analysis:** The user applies filters on the dashboard to isolate a critical pattern (e.g., countries where `PLHIV_0_19` is high but `ART_Coverage_0_14_Pct` is low).
  * **AI-Driven Guidance:** The system immediately sends the quantitative findings to the LLM engine for interpretation.
  * **Insight Generation:** The LLM generates **evidence-based policy priorities** and **strategic recommendations** (the "What to Do").
  * **Zero-Draft Reporting:** The generated output is formatted as a structured narrative, ready to be used as a **zero-draft report** or **presentation slide text**.

-----

## 🛠️ System Architecture and Quick Start

### Prerequisites

  * Python 3.8+
  * An active LLM API key (e.g., Gemini, OpenAI) for the **Policy Guidance Engine**.

### Setup and Execution

Execute these commands sequentially within your project root directory.

#### Environment Setup

Ensure dependency isolation by initializing and activating a clean virtual environment.

```bash
# Deactivate any active environment
deactivate

# Create and activate a new virtual environment
python3 -m venv venv
source venv/bin/activate
```

#### Dependency Installation

Install all required libraries, including Streamlit for the UI and specialized data handling packages.

```bash
# Install all required libraries
pip install -r requirements.txt
```

#### Run ETL Pipeline

Execute the Python script to ingest, cleanse, and harmonize the raw data sources into a single, analysis-ready file (`unicef_hiv_tech.csv`).

```bash
# Execute the Auditable ETL process
python create_hiv_data.py
```

#### Launch Dashboard

Start the interactive Streamlit web application to begin data exploration and automated reporting.

```bash
# Launch the interactive analysis dashboard
streamlit run appunicef.py
```

-----

## III. Core Technical Modules

The framework is built upon robust, version-controlled modules designed for **scalability** and **analytical throughput**.

### 1\. Auditable Data Ingestion & Harmonization Engine

  * **Modular ETL:** Uses Pandas to systematically ingest and **harmonize longitudinal time-series data** from diverse global health sources.
  * **Guaranteed Reproducibility:** All transformation logic (`create_hiv_data.py`) is scripted, ensuring every analytical output is **fully traceable** and reproducible across refreshes.
  * **Indicator Standardization:** Enforces standardization of critical metrics (`ART_Coverage`, `Annual_New_Infections`, `MTCT_Rate`) for robust cross-country and regional comparison.

### 2\. High-Throughput Interactive Analytics Dashboard

  * **Streamlit UI Integration:** Provides a responsive interface for instant strategic filtering by **Region** and **Country**, optimized for identifying service delivery bottlenecks and regional disparities.
  * **Targeted Insight Generation:** Facilitates rapid comparative analysis of key performance indicators to inform resource prioritization.
  * **Scalable Architecture:** Designed to seamlessly integrate future facility-level and geospatial data for advanced subnational monitoring.

### 3\. AI-Driven Policy Guidance Engine (LLM-Integrated)

  * **Data-Grounded Narrative Generation:** Integrates LLMs (e.g., Gemini, GPT) to interpret filtered quantitative results, ensuring all generated narratives are strictly **evidence-based** and contextually relevant.
  * **Automated Policy Guidance:** The LLM automatically generates **actionable policy priorities** and strategic recommendations based on detected data patterns (e.g., trend divergence, coverage gaps).
  * **Output Optimization for Delivery:** Produces structured text optimized for immediate use in **presentation slides** and **executive briefings**, drastically reducing manual reporting effort.
