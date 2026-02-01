"""
ChurnGuard: AI-Powered ARR Risk & Retention Dashboard
======================================================
Production-grade Streamlit dashboard with:
- Dual data ingestion (CSV uploads + API connections)
- Salesforce, Amplitude, Zendesk integrations
- Claude AI-powered churn analysis and retention recommendations
- Dark theme with modern SaaS aesthetics

Author: Claude (Anthropic)
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import os

# =============================================================================
# OPTIONAL IMPORTS (graceful fallback if not installed)
# =============================================================================

# NEW
try:
    from google import genai
    from google.genai import types  # Useful for configuration types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

# Legacy Anthropic support (optional)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Keywords AI / OpenAI-compatible support
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from simple_salesforce import Salesforce
    SALESFORCE_AVAILABLE = True
except ImportError:
    SALESFORCE_AVAILABLE = False
    Salesforce = None

try:
    from zenpy import Zenpy
    ZENDESK_AVAILABLE = True
except ImportError:
    ZENDESK_AVAILABLE = False
    Zenpy = None

import requests  # For Amplitude API

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class Region(Enum):
    """Customer regions for filtering."""
    ALL = "All Regions"
    AMER = "AMER"
    EMEA = "EMEA"
    APAC = "APAC"

@dataclass
class Benchmark:
    """Benchmark configuration for risk metrics."""
    core_module_adoption: float = 0.80
    onboarding_completion: float = 0.90
    weekly_logins: int = 5
    time_to_first_value_days: int = 14
    seat_utilization: float = 0.75
    support_tickets_threshold: int = 3
    nps_score: float = 8.0

# Default benchmarks
DEFAULT_BENCHMARKS = Benchmark()

# Risk category weights
RISK_WEIGHTS = {
    "product": 0.30,
    "process": 0.25,
    "development": 0.25,
    "relationship": 0.20
}

# Color palette
COLORS = {
    "background": "#0e1117",
    "card_bg": "#1a1d24",
    "card_border": "#2d3139",
    "text_primary": "#ffffff",
    "text_secondary": "#8b949e",
    "accent_red": "#ff4d4d",
    "accent_orange": "#ff9f43",
    "accent_yellow": "#feca57",
    "accent_green": "#26de81",
    "accent_blue": "#54a0ff",
    "accent_purple": "#a55eea",
}

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for dark theme and professional styling."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background-color: #0e1117;
            font-family: 'Outfit', sans-serif;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        [data-testid="stSidebar"] {
            background-color: #0a0c10;
            border-right: 1px solid #1f2937;
        }
        
        .risk-card {
            background: linear-gradient(145deg, #1a1d24 0%, #141720 100%);
            border: 1px solid #2d3139;
            border-radius: 16px;
            padding: 20px;
            margin: 8px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }
        
        .risk-card:hover {
            border-color: #3d4149;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
            transform: translateY(-2px);
        }
        
        .risk-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 16px;
        }
        
        .risk-card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .risk-badge-high {
            background: rgba(255, 77, 77, 0.15);
            color: #ff4d4d;
            border: 1px solid rgba(255, 77, 77, 0.3);
        }
        
        .risk-badge-medium {
            background: rgba(255, 159, 67, 0.15);
            color: #ff9f43;
            border: 1px solid rgba(255, 159, 67, 0.3);
        }
        
        .risk-badge-low {
            background: rgba(38, 222, 129, 0.15);
            color: #26de81;
            border: 1px solid rgba(38, 222, 129, 0.3);
        }
        
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #8b949e;
            margin-top: 4px;
        }
        
        .benchmark-text {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 8px;
        }
        
        .score-dots {
            display: flex;
            gap: 4px;
            margin-top: 12px;
        }
        
        .score-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #2d3139;
        }
        
        .score-dot-filled {
            background: #ff9f43;
        }
        
        .sub-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #252830;
            border-radius: 12px;
            padding: 14px;
            margin: 8px 0;
        }
        
        .sub-card-title {
            font-size: 0.9rem;
            font-weight: 500;
            color: #d1d5db;
            margin-bottom: 8px;
        }
        
        .sub-metric {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, #1a1d24 0%, #0e1117 100%);
            border: 1px solid #2d3139;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }
        
        .header-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 8px 0;
        }
        
        .header-subtitle {
            font-size: 1rem;
            color: #8b949e;
        }
        
        .ai-insight-card {
            background: linear-gradient(135deg, #1a1d24 0%, #0f1419 100%);
            border: 1px solid #2d3139;
            border-left: 4px solid #54a0ff;
            border-radius: 12px;
            padding: 20px;
            margin: 12px 0;
        }
        
        .ai-insight-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }
        
        .ai-insight-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        .ai-insight-content {
            color: #d1d5db;
            line-height: 1.6;
        }
        
        .pain-point {
            background: rgba(255, 77, 77, 0.1);
            border-left: 3px solid #ff4d4d;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .recommendation {
            background: rgba(38, 222, 129, 0.1);
            border-left: 3px solid #26de81;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .impact-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .impact-high {
            background: rgba(38, 222, 129, 0.2);
            color: #26de81;
        }
        
        .impact-medium {
            background: rgba(255, 159, 67, 0.2);
            color: #ff9f43;
        }
        
        .connection-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .status-connected {
            background: rgba(38, 222, 129, 0.15);
            color: #26de81;
        }
        
        .status-disconnected {
            background: rgba(255, 77, 77, 0.15);
            color: #ff4d4d;
        }
        
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, #3d4149 50%, transparent 100%);
            margin: 24px 0;
        }
        
        .api-input-container {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid #2d3139;
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
        }
        
        div[data-testid="stExpander"] {
            background-color: #1a1d24;
            border: 1px solid #2d3139;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #8b949e;
            padding: 8px 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(84, 160, 255, 0.15);
            color: #54a0ff;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.4s ease forwards;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .loading-pulse {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# AUTHENTICATION (Simple Password-Based)
# =============================================================================

def check_password() -> bool:
    """
    Simple password authentication.
    For production, use Streamlit-Authenticator or OAuth.
    """
    # Authentication disabled - always allow access
    return True

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def generate_sample_data(n_accounts: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic sample customer data."""
    np.random.seed(seed)

    regions = ["AMER", "EMEA", "APAC"]
    region_weights = [0.45, 0.35, 0.20]
    industries = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Education"]
    company_sizes = ["SMB", "Mid-Market", "Enterprise"]

    data = {
        "account_id": [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)],
        "company_name": [f"Company {chr(65 + i % 26)}{i}" for i in range(n_accounts)],
        "region": np.random.choice(regions, n_accounts, p=region_weights),
        "industry": np.random.choice(industries, n_accounts),
        "company_size": np.random.choice(company_sizes, n_accounts, p=[0.5, 0.35, 0.15]),
        "arr_value": np.random.lognormal(mean=10.5, sigma=1.2, size=n_accounts).astype(int) * 100,
        "contract_start_date": [
            (datetime.now() - timedelta(days=np.random.randint(30, 730))).strftime("%Y-%m-%d")
            for _ in range(n_accounts)
        ],
        "renewal_date": [
            (datetime.now() + timedelta(days=np.random.randint(-30, 365))).strftime("%Y-%m-%d")
            for _ in range(n_accounts)
        ],
    }

    df = pd.DataFrame(data)

    # Generate correlated metrics
    base_health = np.random.beta(5, 2, n_accounts)

    df["core_module_adoption"] = np.clip(base_health * np.random.uniform(0.7, 1.3, n_accounts), 0.15, 1.0)
    df["onboarding_completion_pct"] = np.clip(base_health * np.random.uniform(0.8, 1.2, n_accounts), 0.2, 1.0)
    df["weekly_logins"] = np.clip((base_health * 10 * np.random.uniform(0.5, 1.5, n_accounts)).astype(int), 0, 20)
    df["time_to_first_value_days"] = np.clip(((1 - base_health) * 45 + np.random.normal(0, 5, n_accounts)).astype(int), 3, 90)
    df["seat_utilization_pct"] = np.clip(base_health * np.random.uniform(0.6, 1.2, n_accounts), 0.1, 1.0)
    df["feature_adoption_score"] = np.clip(base_health * np.random.uniform(0.7, 1.2, n_accounts), 0.1, 1.0)
    df["support_tickets_last_quarter"] = np.clip(((1 - base_health) * 8 + np.random.poisson(2, n_accounts)).astype(int), 0, 20)
    df["nps_score"] = np.clip((base_health * 10 + np.random.normal(0, 1.5, n_accounts)), 0, 10)
    df["csm_engagement_score"] = np.clip(base_health * np.random.uniform(0.6, 1.3, n_accounts), 0.2, 1.0)
    df["training_completion_pct"] = np.clip(base_health * np.random.uniform(0.5, 1.2, n_accounts), 0.1, 1.0)
    df["api_integration_count"] = np.clip((base_health * 8 * np.random.uniform(0.3, 1.5, n_accounts)).astype(int), 0, 15)
    df["days_since_last_login"] = np.clip(((1 - base_health) * 30 + np.random.exponential(5, n_accounts)).astype(int), 0, 90)
    df["escalation_count"] = np.clip(((1 - base_health) * 3 + np.random.poisson(0.5, n_accounts)).astype(int), 0, 10)
    df["avg_ticket_resolution_hours"] = np.clip(((1 - base_health) * 48 + np.random.exponential(12, n_accounts)), 1, 168)

    return df

def load_csv_data(salesforce_file, amplitude_file, zendesk_file) -> Optional[pd.DataFrame]:
    """
    Load and merge data from uploaded CSV files.

    Expected columns:
    - Salesforce: account_id, company_name, region, arr_value, industry, company_size
    - Amplitude: account_id, weekly_logins, core_module_adoption, seat_utilization_pct, feature_adoption_score, time_to_first_value_days
    - Zendesk: account_id, support_tickets_last_quarter, escalation_count, avg_ticket_resolution_hours, nps_score
    """
    try:
        # Load each file
        sf_df = pd.read_csv(salesforce_file) if salesforce_file else None
        amp_df = pd.read_csv(amplitude_file) if amplitude_file else None
        zd_df = pd.read_csv(zendesk_file) if zendesk_file else None

        # Start with Salesforce as base (required)
        if sf_df is None:
            st.error("Salesforce data is required as the base dataset")
            return None

        merged_df = sf_df.copy()

        # Merge Amplitude data
        if amp_df is not None and 'account_id' in amp_df.columns:
            # Rename columns if needed (handle sample data column names)
            column_mapping = {
                'integrations_enabled': 'api_integration_count',
            }
            amp_df = amp_df.rename(columns={k: v for k, v in column_mapping.items() if k in amp_df.columns})
            merged_df = merged_df.merge(amp_df, on='account_id', how='left', suffixes=('', '_amp'))

        # Merge Zendesk data
        if zd_df is not None and 'account_id' in zd_df.columns:
            merged_df = merged_df.merge(zd_df, on='account_id', how='left', suffixes=('', '_zd'))

        # Handle duplicate columns from merges (prefer non-suffixed version)
        for col in list(merged_df.columns):
            if col.endswith('_amp') or col.endswith('_zd'):
                base_col = col.rsplit('_', 1)[0]
                if base_col in merged_df.columns:
                    # Fill NaN in base column with values from suffixed column
                    merged_df[base_col] = merged_df[base_col].fillna(merged_df[col])
                    merged_df = merged_df.drop(columns=[col])
                else:
                    # Rename suffixed column to base name
                    merged_df = merged_df.rename(columns={col: base_col})

        # Fill missing values with reasonable defaults
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if merged_df[col].isna().any():
                median_val = merged_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                merged_df[col] = merged_df[col].fillna(median_val)

        return merged_df

    except Exception as e:
        st.error(f"Error loading CSV files: {str(e)}")
        return None

def load_salesforce_data(username: str, password: str, security_token: str, domain: str = 'login') -> Optional[pd.DataFrame]:
    """
    Load account data from Salesforce via API.

    Requires simple-salesforce library.
    """
    if not SALESFORCE_AVAILABLE:
        st.error("simple-salesforce library not installed. Run: pip install simple-salesforce")
        return None

    try:
        sf = Salesforce(
            username=username,
            password=password,
            security_token=security_token,
            domain=domain
        )

        # SOQL query for account data with ARR
        query = """
        SELECT Id, Name, BillingCountry, Industry, AnnualRevenue, 
               NumberOfEmployees, Type, CreatedDate
        FROM Account 
        WHERE AnnualRevenue > 0
        LIMIT 1000
        """

        results = sf.query_all(query)

        if results['totalSize'] == 0:
            st.warning("No accounts found in Salesforce")
            return None

        # Convert to DataFrame
        records = results['records']
        df = pd.DataFrame([{
            'account_id': r['Id'],
            'company_name': r['Name'],
            'region': map_country_to_region(r.get('BillingCountry', '')),
            'industry': r.get('Industry', 'Unknown'),
            'arr_value': r.get('AnnualRevenue', 0),
            'company_size': categorize_company_size(r.get('NumberOfEmployees', 0)),
            'contract_start_date': r.get('CreatedDate', '')[:10] if r.get('CreatedDate') else ''
        } for r in records])

        return df

    except Exception as e:
        st.error(f"Salesforce connection error: {str(e)}")
        return None

def load_amplitude_data(api_key: str, secret_key: str, start_date: str = None) -> Optional[pd.DataFrame]:
    """
    Load product usage data from Amplitude via API.

    Note: This uses the Amplitude Export API. Adjust endpoints based on your Amplitude setup.
    """
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        # Amplitude Export API endpoint (adjust for your project)
        url = f"https://amplitude.com/api/2/export"

        params = {
            'start': f"{start_date}T00",
            'end': f"{end_date}T23"
        }

        response = requests.get(
            url,
            auth=(api_key, secret_key),
            params=params,
            timeout=60
        )

        if response.status_code == 200:
            # Parse response (format depends on Amplitude setup)
            # This is a simplified example - actual implementation depends on your event structure
            data = response.json() if response.headers.get('content-type') == 'application/json' else []

            if not data:
                st.warning("No data returned from Amplitude. Using sample data.")
                return None

            # Process and aggregate by account
            # (Implementation depends on your Amplitude event structure)
            df = pd.DataFrame(data)
            return df

        elif response.status_code == 401:
            st.error("Amplitude authentication failed. Check your API credentials.")
            return None
        else:
            st.error(f"Amplitude API error: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Amplitude connection error: {str(e)}")
        return None

def load_zendesk_data(subdomain: str, email: str, api_token: str) -> Optional[pd.DataFrame]:
    """
    Load support ticket data from Zendesk via API.

    Requires zenpy library.
    """
    if not ZENDESK_AVAILABLE:
        st.error("zenpy library not installed. Run: pip install zenpy")
        return None

    try:
        creds = {
            'email': email,
            'token': api_token,
            'subdomain': subdomain
        }

        zenpy_client = Zenpy(**creds)

        # Get tickets from last 90 days
        start_date = datetime.now() - timedelta(days=90)

        tickets = []
        for ticket in zenpy_client.search(created_greater_than=start_date, type='ticket'):
            tickets.append({
                'ticket_id': ticket.id,
                'account_id': ticket.organization_id if ticket.organization else None,
                'status': ticket.status,
                'priority': ticket.priority,
                'created_at': ticket.created_at,
                'updated_at': ticket.updated_at,
                'satisfaction_rating': getattr(ticket.satisfaction_rating, 'score', None) if ticket.satisfaction_rating else None
            })

        df = pd.DataFrame(tickets)

        if df.empty:
            st.warning("No tickets found in Zendesk")
            return None

        # Aggregate by account
        account_stats = df.groupby('account_id').agg({
            'ticket_id': 'count',
            'priority': lambda x: (x == 'urgent').sum() + (x == 'high').sum(),
            'satisfaction_rating': 'mean'
        }).reset_index()

        account_stats.columns = ['account_id', 'support_tickets_last_quarter', 'escalation_count', 'nps_score']

        return account_stats

    except Exception as e:
        st.error(f"Zendesk connection error: {str(e)}")
        return None

def merge_datasets(sf_df: pd.DataFrame, amp_df: Optional[pd.DataFrame], zd_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge data from multiple sources into unified dataset.
    """
    merged = sf_df.copy()

    if amp_df is not None and not amp_df.empty and 'account_id' in amp_df.columns:
        merged = merged.merge(amp_df, on='account_id', how='left', suffixes=('', '_amp'))

    if zd_df is not None and not zd_df.empty and 'account_id' in zd_df.columns:
        merged = merged.merge(zd_df, on='account_id', how='left', suffixes=('', '_zd'))

    # Fill missing numeric values
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    return merged

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def map_country_to_region(country: str) -> str:
    """Map country to region."""
    if not country:
        return "AMER"

    country = country.upper()

    apac_countries = ['CHINA', 'JAPAN', 'INDIA', 'AUSTRALIA', 'SINGAPORE', 'KOREA', 'HONG KONG', 'TAIWAN']
    emea_countries = ['UNITED KINGDOM', 'UK', 'GERMANY', 'FRANCE', 'ITALY', 'SPAIN', 'NETHERLANDS',
                      'SWEDEN', 'NORWAY', 'DENMARK', 'SWITZERLAND', 'AUSTRIA', 'BELGIUM', 'IRELAND',
                      'SOUTH AFRICA', 'UAE', 'ISRAEL', 'SAUDI ARABIA']

    if any(c in country for c in apac_countries):
        return "APAC"
    elif any(c in country for c in emea_countries):
        return "EMEA"
    return "AMER"

def categorize_company_size(employees: int) -> str:
    """Categorize company by employee count."""
    if employees < 100:
        return "SMB"
    elif employees < 1000:
        return "Mid-Market"
    return "Enterprise"

def format_currency(value: float) -> str:
    """Format value as currency."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    return f"${value:.0f}"

# =============================================================================
# RISK COMPUTATION
# =============================================================================

def compute_risk_scores(df: pd.DataFrame, benchmarks: Benchmark) -> pd.DataFrame:
    """
    Compute risk scores for each account.
    """
    df = df.copy()

    # Ensure required columns exist with defaults
    required_cols = {
        'core_module_adoption': 0.5,
        'onboarding_completion_pct': 0.5,
        'weekly_logins': 3,
        'time_to_first_value_days': 21,
        'seat_utilization_pct': 0.5,
        'support_tickets_last_quarter': 2,
        'nps_score': 7.0,
        'csm_engagement_score': 0.5,
        'training_completion_pct': 0.5,
        'feature_adoption_score': 0.5,
        'api_integration_count': 2
    }

    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default

    # Calculate deviation scores (higher = worse)
    # Use np.maximum for element-wise max with pandas Series
    df['product_risk_score'] = (
        np.maximum(0, (benchmarks.core_module_adoption - df['core_module_adoption'].clip(0, 1))) * 0.4 +
        np.maximum(0, (benchmarks.seat_utilization - df['seat_utilization_pct'].clip(0, 1))) * 0.3 +
        np.maximum(0, (0.7 - df['feature_adoption_score'].clip(0, 1))) * 0.3
    )

    df['process_risk_score'] = (
        np.maximum(0, (benchmarks.onboarding_completion - df['onboarding_completion_pct'].clip(0, 1))) * 0.35 +
        np.clip((benchmarks.weekly_logins - df['weekly_logins']) / benchmarks.weekly_logins, 0, 1) * 0.35 +
        np.clip((df['time_to_first_value_days'] - benchmarks.time_to_first_value_days) / 30, 0, 1) * 0.30
    )

    df['development_risk_score'] = (
        np.maximum(0, (0.8 - df['training_completion_pct'].clip(0, 1))) * 0.5 +
        np.clip((5 - df['api_integration_count']) / 5, 0, 1) * 0.5
    )

    df['relationship_risk_score'] = (
        np.clip((benchmarks.nps_score - df['nps_score']) / 10, 0, 1) * 0.35 +
        np.maximum(0, (0.8 - df['csm_engagement_score'].clip(0, 1))) * 0.35 +
        np.clip(df['support_tickets_last_quarter'] / 10, 0, 1) * 0.30
    )

    # Overall churn probability
    df['churn_probability'] = (
        df['product_risk_score'] * RISK_WEIGHTS['product'] +
        df['process_risk_score'] * RISK_WEIGHTS['process'] +
        df['development_risk_score'] * RISK_WEIGHTS['development'] +
        df['relationship_risk_score'] * RISK_WEIGHTS['relationship']
    )

    # Normalize to 0-1 range
    df['churn_probability'] = df['churn_probability'].clip(0, 1)

    # Risk tier assignment
    df['risk_tier'] = pd.cut(
        df['churn_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return df

def compute_risk_attribution(df: pd.DataFrame, benchmarks: Benchmark) -> Dict:
    """
    Compute aggregated risk attribution by category.
    """
    at_risk = df[df['risk_tier'].isin(['High', 'Medium'])]
    total_risk_arr = at_risk['arr_value'].sum() if 'arr_value' in at_risk.columns else 0

    # Category scores
    product_risk = df['product_risk_score'].mean() if 'product_risk_score' in df.columns else 0.3
    process_risk = df['process_risk_score'].mean() if 'process_risk_score' in df.columns else 0.25
    development_risk = df['development_risk_score'].mean() if 'development_risk_score' in df.columns else 0.25
    relationship_risk = df['relationship_risk_score'].mean() if 'relationship_risk_score' in df.columns else 0.2

    total_raw_risk = product_risk + process_risk + development_risk + relationship_risk

    if total_raw_risk > 0:
        product_pct = (product_risk / total_raw_risk) * 100
        process_pct = (process_risk / total_raw_risk) * 100
        development_pct = (development_risk / total_raw_risk) * 100
        relationship_pct = (relationship_risk / total_raw_risk) * 100
    else:
        product_pct = process_pct = development_pct = relationship_pct = 25

    return {
        "total_risk_arr": total_risk_arr,
        "at_risk_accounts": len(at_risk),
        "high_risk_accounts": len(df[df['risk_tier'] == 'High']),
        "total_accounts": len(df),
        "categories": {
            "product": {
                "percentage": round(product_pct),
                "raw_score": product_risk,
                "health_score": int(5 - min(5, product_risk * 10)),
                "sub_metrics": {
                    "product_gaps": {
                        "label": "Core Module Adoption",
                        "value": round(df['core_module_adoption'].mean() * 100),
                        "benchmark": int(benchmarks.core_module_adoption * 100),
                        "delta": round((df['core_module_adoption'].mean() - benchmarks.core_module_adoption) * 100),
                    },
                    "seat_utilization": {
                        "label": "Seat Utilization",
                        "value": round(df['seat_utilization_pct'].mean() * 100),
                        "benchmark": int(benchmarks.seat_utilization * 100),
                        "delta": round((df['seat_utilization_pct'].mean() - benchmarks.seat_utilization) * 100),
                    },
                    "feature_adoption": {
                        "label": "Feature Adoption",
                        "value": round(df['feature_adoption_score'].mean() * 100),
                        "benchmark": 70,
                        "delta": round((df['feature_adoption_score'].mean() - 0.7) * 100),
                    }
                }
            },
            "process": {
                "percentage": round(process_pct),
                "raw_score": process_risk,
                "health_score": int(5 - min(5, process_risk * 10)),
                "sub_metrics": {
                    "onboarding": {
                        "label": "Onboarding Completion",
                        "value": round(df['onboarding_completion_pct'].mean() * 100),
                        "benchmark": int(benchmarks.onboarding_completion * 100),
                        "delta": round((df['onboarding_completion_pct'].mean() - benchmarks.onboarding_completion) * 100),
                    },
                    "login_frequency": {
                        "label": "Weekly Logins (Avg)",
                        "value": round(df['weekly_logins'].mean(), 1),
                        "benchmark": benchmarks.weekly_logins,
                        "delta": round(df['weekly_logins'].mean() - benchmarks.weekly_logins, 1),
                    },
                    "ttfv": {
                        "label": "Time to First Value (Days)",
                        "value": round(df['time_to_first_value_days'].mean()),
                        "benchmark": benchmarks.time_to_first_value_days,
                        "delta": round(df['time_to_first_value_days'].mean() - benchmarks.time_to_first_value_days),
                        "inverse": True
                    }
                }
            },
            "development": {
                "percentage": round(development_pct),
                "raw_score": development_risk,
                "health_score": int(5 - min(5, development_risk * 10)),
                "sub_metrics": {
                    "training": {
                        "label": "Training Completion",
                        "value": round(df['training_completion_pct'].mean() * 100),
                        "benchmark": 80,
                        "delta": round((df['training_completion_pct'].mean() - 0.8) * 100),
                    },
                    "api_integration": {
                        "label": "API Integrations (Avg)",
                        "value": round(df['api_integration_count'].mean(), 1),
                        "benchmark": 5,
                        "delta": round(df['api_integration_count'].mean() - 5, 1),
                    }
                }
            },
            "relationship": {
                "percentage": round(relationship_pct),
                "raw_score": relationship_risk,
                "health_score": int(5 - min(5, relationship_risk * 10)),
                "sub_metrics": {
                    "nps": {
                        "label": "NPS Score",
                        "value": round(df['nps_score'].mean(), 1),
                        "benchmark": benchmarks.nps_score,
                        "delta": round(df['nps_score'].mean() - benchmarks.nps_score, 1),
                    },
                    "csm_engagement": {
                        "label": "CSM Engagement",
                        "value": round(df['csm_engagement_score'].mean() * 100),
                        "benchmark": 80,
                        "delta": round((df['csm_engagement_score'].mean() - 0.8) * 100),
                    },
                    "support_health": {
                        "label": "Support Tickets (Avg)",
                        "value": round(df['support_tickets_last_quarter'].mean(), 1),
                        "benchmark": benchmarks.support_tickets_threshold,
                        "delta": round(df['support_tickets_last_quarter'].mean() - benchmarks.support_tickets_threshold, 1),
                        "inverse": True
                    }
                }
            }
        }
    }

# =============================================================================
# CLAUDE AI ANALYSIS
# =============================================================================

def prepare_analysis_summary(df: pd.DataFrame, risk_data: Dict) -> str:
    """
    Prepare a rich summary for Claude analysis.
    """
    # Regional breakdown
    regional_stats = df.groupby('region').agg({
        'account_id': 'count',
        'arr_value': 'sum',
        'churn_probability': 'mean',
        'core_module_adoption': 'mean',
        'weekly_logins': 'mean',
        'support_tickets_last_quarter': 'mean',
        'nps_score': 'mean'
    }).round(2).to_dict()

    # Risk tier breakdown
    risk_breakdown = df.groupby('risk_tier').agg({
        'account_id': 'count',
        'arr_value': 'sum'
    }).to_dict()

    # Top risk factors
    high_risk = df[df['risk_tier'] == 'High']

    summary = {
        "overview": {
            "total_accounts": len(df),
            "total_arr": int(df['arr_value'].sum()),
            "at_risk_accounts": risk_data['at_risk_accounts'],
            "high_risk_accounts": risk_data['high_risk_accounts'],
            "arr_at_risk": int(risk_data['total_risk_arr']),
            "average_churn_probability": round(df['churn_probability'].mean() * 100, 1)
        },
        "risk_attribution": {
            "product_risk_pct": risk_data['categories']['product']['percentage'],
            "process_risk_pct": risk_data['categories']['process']['percentage'],
            "development_risk_pct": risk_data['categories']['development']['percentage'],
            "relationship_risk_pct": risk_data['categories']['relationship']['percentage']
        },
        "key_metrics": {
            "avg_core_module_adoption": round(df['core_module_adoption'].mean() * 100, 1),
            "avg_onboarding_completion": round(df['onboarding_completion_pct'].mean() * 100, 1),
            "avg_weekly_logins": round(df['weekly_logins'].mean(), 1),
            "avg_time_to_first_value_days": round(df['time_to_first_value_days'].mean(), 1),
            "avg_seat_utilization": round(df['seat_utilization_pct'].mean() * 100, 1),
            "avg_nps_score": round(df['nps_score'].mean(), 1),
            "avg_support_tickets": round(df['support_tickets_last_quarter'].mean(), 1),
            "avg_training_completion": round(df['training_completion_pct'].mean() * 100, 1)
        },
        "regional_breakdown": regional_stats,
        "benchmark_gaps": {
            "core_adoption_gap": round((0.80 - df['core_module_adoption'].mean()) * 100, 1),
            "onboarding_gap": round((0.90 - df['onboarding_completion_pct'].mean()) * 100, 1),
            "seat_utilization_gap": round((0.75 - df['seat_utilization_pct'].mean()) * 100, 1),
            "nps_gap": round(8.0 - df['nps_score'].mean(), 1)
        },
        "high_risk_characteristics": {
            "count": len(high_risk),
            "avg_adoption": round(high_risk['core_module_adoption'].mean() * 100, 1) if len(high_risk) > 0 else 0,
            "avg_support_tickets": round(high_risk['support_tickets_last_quarter'].mean(), 1) if len(high_risk) > 0 else 0,
            "total_arr": int(high_risk['arr_value'].sum()) if len(high_risk) > 0 else 0
        }
    }

    return json.dumps(summary, indent=2)

def get_ai_insights(summary: str, api_key: str = None, provider: str = "google") -> Optional[str]:
    """
    Get AI-powered retention insights from Google Gemini or Anthropic Claude.

    Args:
        summary: JSON summary of customer data
        api_key: API key for the AI provider
        provider: "google" for Gemini or "anthropic" for Claude
    """

    # Common prompt for both providers
    prompt = f"""You are an elite Customer Success and Retention expert with deep expertise in SaaS metrics, churn analysis, and data-driven retention strategies.

Analyze the following aggregated customer churn risk data from our integrated Salesforce (CRM), Amplitude (product analytics), and Zendesk (support) systems:

```json
{summary}
```

Provide a comprehensive retention analysis with the following structure:

## 🔴 Top Pain Points Driving ARR Risk

Identify 4-6 critical pain points causing churn risk. For each:
- Clear problem statement
- Supporting data/evidence from the metrics
- Estimated ARR impact if unaddressed

## 🔍 Root Cause Analysis

For each pain point, explain:
- The underlying root cause
- How different data sources confirm this (e.g., low adoption + high tickets = product-market fit issues)
- Customer segments most affected

## ✅ Prioritized Retention Recommendations

Provide 6-8 specific, actionable recommendations ranked by estimated impact:
1. **[Action Title]** - [IMPACT: High/Medium]
   - What to do (specific steps)
   - Expected outcome
   - Estimated ARR lift potential (% or range)
   - Timeline to implement

Focus on actionable insights that a CS team could implement immediately. Use concrete numbers from the data to support your analysis. Be direct and specific - avoid generic advice."""

    # Try Keywords AI first (most reliable)
    if provider == "keywords" or (provider == "auto" and OPENAI_AVAILABLE):
        return _get_keywords_ai_insights(prompt, api_key)

    # Try Google Gemini
    elif provider == "google" and GOOGLE_AI_AVAILABLE:
        return _get_gemini_insights(prompt, api_key)

    # Fallback to Anthropic Claude
    elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
        return _get_claude_insights(prompt, api_key)

    # No provider available
    return _get_fallback_insights()


def _get_keywords_ai_insights(prompt: str, api_key: str = None) -> str:
    """Get insights from Keywords AI."""
    import requests as req
    
    if not api_key:
        try:
            api_key = st.secrets.get("KEYWORDS_API_KEY")
        except:
            pass

    if not api_key:
        api_key = os.environ.get("KEYWORDS_API_KEY")

    if not api_key:
        return "❌ **Keywords AI API key not configured.**"

    # Try multiple models in order of preference
    models_to_try = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022", 
        "gpt-4o-mini",
        "gpt-4o",
        "gemini-1.5-flash",
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    
    last_error = None
    
    for model in models_to_try:
        try:
            data = {
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 2500,
                'temperature': 0.7
            }
            
            response = req.post(
                'https://api.keywordsai.co/api/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                last_error = response.text
                # If it's an auth/credentials error, try the next model
                if response.status_code in [401, 403]:
                    continue
                # For other errors, return immediately
                return f"❌ **Error with Keywords AI ({model}):** {response.text}"
                
        except Exception as e:
            last_error = str(e)
            continue
    
    return f"❌ **Keywords AI: No working model found.** You may need to add provider credentials in your Keywords AI dashboard.\n\nLast error: {last_error}"


def _get_gemini_insights(prompt: str, api_key: str = None) -> str:
    """Get insights from Google Gemini using the new 2025 SDK."""
    if not GOOGLE_AI_AVAILABLE:
        return "## ⚠️ Google AI Library Not Available\n\nRun: `pip install google-genai`"

    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except:
            pass

    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        return "❌ **Google API key not configured.**"

    try:
        # 1. Instantiate the Client (Replaces genai.configure)
        client = genai.Client(api_key=api_key)

        # 2. Use the models.generate_content method
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2500,
                temperature=0.7,
            )
        )

        return response.text

    except Exception as e:
        # Standardize error handling for the new client
        return f"❌ **Error getting AI insights:** {str(e)}"

def _get_claude_insights(prompt: str, api_key: str = None) -> str:
    """Get insights from Anthropic Claude (fallback)."""
    if not ANTHROPIC_AVAILABLE:
        return "❌ **Anthropic library not installed.** Run: `pip install anthropic`"

    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return "❌ **Anthropic API key not configured.**"

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text

    except Exception as e:
        return f"❌ **Error with Claude:** {str(e)}"


def _get_fallback_insights() -> str:
    """Return static fallback insights when no AI provider is available."""
    return """
## ⚠️ AI Provider Not Configured

No AI library is available. To enable AI-powered insights, install one of:

**Google Gemini (Recommended):**
```bash
pip install google-generativeai
```
Then add `GOOGLE_API_KEY` to your secrets.

**Anthropic Claude (Alternative):**
```bash
pip install anthropic
```
Then add `ANTHROPIC_API_KEY` to your secrets.

---

### Sample Analysis (Static)

Based on typical patterns in customer success data:

**🔴 Top Pain Points:**
1. **Low Core Module Adoption** - Customers aren't discovering full product value, leading to perceived low ROI
2. **Extended Time-to-First-Value** - Onboarding friction delays the "aha moment"
3. **High Support Ticket Volume** - Indicates product complexity or quality issues
4. **Poor Seat Utilization** - Customers paying for unused capacity

**✅ Recommendations:**
1. **Implement Guided Product Tours** [HIGH IMPACT] - Interactive walkthroughs for underutilized features
2. **Create Quick-Start Templates** [HIGH IMPACT] - Accelerate time-to-value with pre-built configurations
3. **Proactive CSM Outreach** [MEDIUM IMPACT] - Trigger alerts for accounts showing risk signals
4. **Usage-Based Pricing Review** [MEDIUM IMPACT] - Align pricing with actual consumption patterns
"""

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_score_dots(score: int, max_score: int = 5, filled_color: str = "#ff9f43") -> str:
    """Render score as colored dots."""
    dots = []
    for i in range(max_score):
        if i < score:
            dots.append(f'<span class="score-dot score-dot-filled" style="background: {filled_color};"></span>')
        else:
            dots.append('<span class="score-dot"></span>')
    return f'<div class="score-dots">{"".join(dots)}</div>'

def get_risk_badge_class(percentage: int) -> str:
    """Get CSS class for risk badge."""
    if percentage >= 35:
        return "risk-badge-high"
    elif percentage >= 25:
        return "risk-badge-medium"
    return "risk-badge-low"

def render_risk_category_card(category_name: str, category_data: Dict, icon: str) -> None:
    """Render a risk category card."""
    pct = category_data["percentage"]
    health = category_data["health_score"]
    badge_class = get_risk_badge_class(pct)

    if health >= 4:
        dot_color = "#26de81"
    elif health >= 3:
        dot_color = "#ff9f43"
    else:
        dot_color = "#ff4d4d"

    card_html = f'''
    <div class="risk-card animate-fade-in">
        <div class="risk-card-header">
            <p class="risk-card-title">{icon} {category_name}</p>
            <span class="risk-badge {badge_class}">{pct}%</span>
        </div>
        <p class="benchmark-text">Responsible for {pct}% of total risk</p>
        {render_score_dots(health, 5, dot_color)}
        <p class="metric-label" style="margin-top: 4px;">{health}/5 Health Score</p>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander(f"View {category_name} Details"):
        for metric_key, metric_data in category_data["sub_metrics"].items():
            is_inverse = metric_data.get("inverse", False)
            delta = metric_data["delta"]

            if is_inverse:
                delta_class = "metric-delta-negative" if delta > 0 else "metric-delta-positive"
                status_text = "above benchmark ⚠️" if delta > 0 else "at benchmark ✓"
            else:
                delta_class = "metric-delta-negative" if delta < 0 else "metric-delta-positive"
                status_text = "below benchmark ⚠️" if delta < 0 else "at benchmark ✓"

            delta_sign = "+" if delta > 0 else ""

            sub_html = f'''
            <div class="sub-card">
                <p class="sub-card-title">{metric_data["label"]}</p>
                <p class="sub-metric">
                    {metric_data["value"]}
                    <span class="metric-delta {delta_class}">{delta_sign}{delta}</span>
                </p>
                <p class="benchmark-text">Benchmark: {metric_data["benchmark"]} • {status_text}</p>
            </div>
            '''
            st.markdown(sub_html, unsafe_allow_html=True)

def render_header(risk_data: Dict, region: str, data_source: str) -> None:
    """Render the dashboard header."""
    total_risk_arr = risk_data["total_risk_arr"]
    at_risk = risk_data["at_risk_accounts"]
    high_risk = risk_data["high_risk_accounts"]
    total = risk_data["total_accounts"]

    source_badge = "📁 CSV" if data_source == "csv" else "🔌 API" if data_source == "api" else "🎲 Sample"

    header_html = f'''
    <div class="dashboard-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
            <div>
                <h1 class="header-title">🎯 Root-Cause Analysis</h1>
                <p class="header-subtitle">
                    <span style="background: rgba(84, 160, 255, 0.15); color: #54a0ff; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin-right: 8px;">{source_badge}</span>
                    {len(risk_data["categories"])} risk categories • 
                    <span style="color: #ff4d4d;">{high_risk} high-risk</span> / {total} accounts • 
                    Region: <strong>{region}</strong>
                </p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <span style="background: rgba(255, 159, 67, 0.15); color: #ff9f43; padding: 6px 12px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;">
                    +{high_risk} critical
                </span>
            </div>
        </div>
        <div style="margin-top: 20px; display: flex; gap: 40px; flex-wrap: wrap;">
            <div>
                <p class="metric-value" style="color: #ff4d4d;">{format_currency(total_risk_arr)}</p>
                <p class="metric-label">ARR at Risk</p>
            </div>
            <div>
                <p class="metric-value">{at_risk}</p>
                <p class="metric-label">At-Risk Accounts</p>
            </div>
            <div>
                <p class="metric-value" style="color: #ff9f43;">{high_risk}</p>
                <p class="metric-label">High Priority</p>
            </div>
            <div>
                <p class="metric-value" style="color: #26de81;">{total - at_risk}</p>
                <p class="metric-label">Healthy Accounts</p>
            </div>
        </div>
    </div>
    '''
    st.markdown(header_html, unsafe_allow_html=True)

def render_ai_insights_section(df: pd.DataFrame, risk_data: Dict) -> None:
    """Render the AI-powered insights section."""
    st.markdown("### 🤖 AI-Powered Retention Analysis")
    st.markdown("""
    <p style="color: #8b949e; margin-bottom: 20px;">
        Powered by Keywords AI • Analyzes merged Salesforce, Amplitude, and Zendesk data
    </p>
    """, unsafe_allow_html=True)

    # AI Provider selection
    col_provider, col_key, col_btn = st.columns([1, 2, 1])

    with col_provider:
        ai_provider = st.selectbox(
            "AI Provider",
            ["Keywords AI", "Google Gemini", "Anthropic Claude"],
            index=0,
            help="Select your AI provider"
        )

    # Determine which API key to use
    if "Keywords" in ai_provider:
        provider = "keywords"
        key_name = "KEYWORDS_API_KEY"
        key_placeholder = "sk-..."
    elif "Google" in ai_provider:
        provider = "google"
        key_name = "GOOGLE_API_KEY"
        key_placeholder = "AIza..."
    else:
        provider = "anthropic"
        key_name = "ANTHROPIC_API_KEY"
        key_placeholder = "sk-ant-..."

    # Try to get API key from secrets first
    api_key = None
    try:
        api_key = st.secrets.get(key_name)
    except:
        pass

    if not api_key:
        api_key = st.session_state.get(f"{provider}_api_key")

    with col_key:
        if not api_key:
            new_key = st.text_input(
                f"{ai_provider} API Key",
                type="password",
                placeholder=key_placeholder,
                help=f"Enter your {ai_provider} API key to enable AI analysis"
            )
            if new_key:
                st.session_state[f"{provider}_api_key"] = new_key
                api_key = new_key

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔄 Generate AI Analysis", use_container_width=True, type="primary")

    # Show API key help
    if not api_key:
        if provider == "keywords":
            st.info("💡 Get your Keywords AI API key at [Keywords AI Dashboard](https://keywordsai.co)")
        elif provider == "google":
            st.info("💡 Get your free Google API key at [Google AI Studio](https://makersuite.google.com/app/apikey)")
        else:
            st.info("💡 Get your Anthropic API key at [console.anthropic.com](https://console.anthropic.com)")

    if analyze_button or st.session_state.get("ai_insights"):
        if analyze_button:
            with st.spinner(f"🧠 {ai_provider} is analyzing your customer data..."):
                summary = prepare_analysis_summary(df, risk_data)
                insights = get_ai_insights(summary, api_key, provider)
                st.session_state.ai_insights = insights
                st.session_state.ai_summary = summary
                st.session_state.ai_provider_used = ai_provider

        insights = st.session_state.get("ai_insights", "")
        provider_used = st.session_state.get("ai_provider_used", ai_provider)

        if insights:
            st.markdown(f"""
            <div class="ai-insight-card">
                <div class="ai-insight-header">
                    <span style="font-size: 1.5rem;">🧠</span>
                    <span class="ai-insight-title">{provider_used}'s Retention Analysis</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(insights)

            # Show raw data summary in expander
            with st.expander("📊 View Data Summary Sent to AI"):
                st.code(st.session_state.get("ai_summary", ""), language="json")

def render_data_source_selector() -> Tuple[str, Optional[pd.DataFrame]]:
    """Render data source selection UI and return loaded data."""
    st.markdown("### 📥 Data Source")

    source_mode = st.radio(
        "Choose data ingestion method:",
        ["🎲 Sample Data (Demo)", "📁 Upload CSV Files", "🔌 Connect APIs"],
        horizontal=True,
        label_visibility="collapsed"
    )

    df = None
    data_source = "sample"

    if "Sample Data" in source_mode:
        st.info("Using simulated customer data for demonstration. Connect real data sources for production use.")
        if st.button("🔄 Regenerate Sample Data"):
            st.session_state.data_seed = np.random.randint(1, 10000)
        seed = st.session_state.get("data_seed", 42)
        df = generate_sample_data(n_accounts=300, seed=seed)
        data_source = "sample"

    elif "Upload CSV" in source_mode:
        st.markdown("""
        <div class="api-input-container">
            <p style="color: #d1d5db; margin-bottom: 12px;">Upload your data files (CSV format)</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Salesforce Accounts** (Required)")
            sf_file = st.file_uploader("Account data with ARR", type=['csv'], key="sf_upload", label_visibility="collapsed")

        with col2:
            st.markdown("**Amplitude Usage** (Optional)")
            amp_file = st.file_uploader("Product usage metrics", type=['csv'], key="amp_upload", label_visibility="collapsed")

        with col3:
            st.markdown("**Zendesk Tickets** (Optional)")
            zd_file = st.file_uploader("Support ticket data", type=['csv'], key="zd_upload", label_visibility="collapsed")

        if sf_file:
            df = load_csv_data(sf_file, amp_file, zd_file)
            data_source = "csv"
        else:
            st.warning("Please upload at least the Salesforce accounts CSV to proceed.")

        with st.expander("📋 Expected CSV Formats"):
            st.markdown("""
            **Salesforce CSV columns:**
            `account_id, company_name, region, arr_value, industry, company_size, contract_start_date, renewal_date`
            
            **Amplitude CSV columns:**
            `account_id, weekly_logins, core_module_adoption, seat_utilization_pct, feature_adoption_score, time_to_first_value_days, onboarding_completion_pct, training_completion_pct`
            
            **Zendesk CSV columns:**
            `account_id, support_tickets_last_quarter, escalation_count, avg_ticket_resolution_hours, nps_score, csm_engagement_score`
            """)

    elif "Connect APIs" in source_mode:
        st.markdown("""
        <div class="api-input-container">
            <p style="color: #d1d5db; margin-bottom: 12px;">🔐 API credentials are stored in session only (not persisted)</p>
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(["Salesforce", "Amplitude", "Zendesk"])

        sf_data = None
        amp_data = None
        zd_data = None

        with tabs[0]:
            st.markdown("#### Salesforce Connection")
            col1, col2 = st.columns(2)
            with col1:
                sf_username = st.text_input("Username", key="sf_user", placeholder="user@company.com")
                sf_password = st.text_input("Password", type="password", key="sf_pass")
            with col2:
                sf_token = st.text_input("Security Token", type="password", key="sf_token")
                sf_domain = st.selectbox("Domain", ["login", "test"], key="sf_domain")

            if st.button("Connect to Salesforce", key="sf_connect"):
                if all([sf_username, sf_password, sf_token]):
                    with st.spinner("Connecting to Salesforce..."):
                        sf_data = load_salesforce_data(sf_username, sf_password, sf_token, sf_domain)
                        if sf_data is not None:
                            st.session_state.sf_data = sf_data
                            st.success(f"✅ Connected! Loaded {len(sf_data)} accounts")
                else:
                    st.warning("Please fill all Salesforce credentials")

            if st.session_state.get("sf_data") is not None:
                st.markdown('<span class="connection-status status-connected">● Connected</span>', unsafe_allow_html=True)
                sf_data = st.session_state.sf_data

        with tabs[1]:
            st.markdown("#### Amplitude Connection")
            col1, col2 = st.columns(2)
            with col1:
                amp_api_key = st.text_input("API Key", type="password", key="amp_key")
            with col2:
                amp_secret = st.text_input("Secret Key", type="password", key="amp_secret")

            if st.button("Connect to Amplitude", key="amp_connect"):
                if all([amp_api_key, amp_secret]):
                    with st.spinner("Connecting to Amplitude..."):
                        amp_data = load_amplitude_data(amp_api_key, amp_secret)
                        if amp_data is not None:
                            st.session_state.amp_data = amp_data
                            st.success(f"✅ Connected!")
                else:
                    st.warning("Please fill all Amplitude credentials")

            if st.session_state.get("amp_data") is not None:
                st.markdown('<span class="connection-status status-connected">● Connected</span>', unsafe_allow_html=True)
                amp_data = st.session_state.amp_data

        with tabs[2]:
            st.markdown("#### Zendesk Connection")
            col1, col2 = st.columns(2)
            with col1:
                zd_subdomain = st.text_input("Subdomain", key="zd_subdomain", placeholder="yourcompany")
                zd_email = st.text_input("Email", key="zd_email", placeholder="admin@company.com")
            with col2:
                zd_token = st.text_input("API Token", type="password", key="zd_token")

            if st.button("Connect to Zendesk", key="zd_connect"):
                if all([zd_subdomain, zd_email, zd_token]):
                    with st.spinner("Connecting to Zendesk..."):
                        zd_data = load_zendesk_data(zd_subdomain, zd_email, zd_token)
                        if zd_data is not None:
                            st.session_state.zd_data = zd_data
                            st.success(f"✅ Connected!")
                else:
                    st.warning("Please fill all Zendesk credentials")

            if st.session_state.get("zd_data") is not None:
                st.markdown('<span class="connection-status status-connected">● Connected</span>', unsafe_allow_html=True)
                zd_data = st.session_state.zd_data

        # Merge available data
        sf_data = st.session_state.get("sf_data")
        amp_data = st.session_state.get("amp_data")
        zd_data = st.session_state.get("zd_data")

        if sf_data is not None:
            df = merge_datasets(sf_data, amp_data, zd_data)
            data_source = "api"
        else:
            st.info("Connect to Salesforce (required) to load account data. Amplitude and Zendesk are optional.")

    return data_source, df

def render_sidebar(df: Optional[pd.DataFrame]) -> Tuple[str, Benchmark]:
    """Render sidebar navigation and controls."""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 20px 0; border-bottom: 1px solid #2d3139; margin-bottom: 20px;">
            <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.4rem; font-weight: 700; color: #ffffff; margin: 0;">
                🛡️ ChurnGuard
            </h2>
            <p style="color: #6b7280; font-size: 0.85rem; margin-top: 4px;">AI-Powered Retention Analytics</p>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown("### Navigation")
        nav_options = ["📊 Overview", "🤖 AI Insights", "📈 Revenue Impact", "🎯 Benchmarking", "📋 Accounts"]

        if "nav_selection" not in st.session_state:
            st.session_state.nav_selection = "📊 Overview"

        for nav in nav_options:
            is_selected = st.session_state.nav_selection == nav
            if st.button(nav, key=f"nav_{nav}", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.nav_selection = nav

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Region Filter
        st.markdown("### 🌍 Region Filter")
        regions = ["All Regions"]
        if df is not None and 'region' in df.columns:
            regions += df['region'].unique().tolist()

        region = st.selectbox("Select Region", options=regions, label_visibility="collapsed")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Benchmark Adjustments
        st.markdown("### ⚙️ Benchmarks")

        with st.expander("Adjust Thresholds"):
            core_adoption = st.slider("Core Adoption %", 50, 100, 80) / 100
            onboarding = st.slider("Onboarding %", 50, 100, 90) / 100
            weekly_logins = st.slider("Weekly Logins", 1, 10, 5)
            ttfv = st.slider("TTFV (days)", 7, 30, 14)
            seat_util = st.slider("Seat Utilization %", 50, 100, 75) / 100
            nps = st.slider("NPS Score", 5.0, 10.0, 8.0, 0.5)

        benchmarks = Benchmark(
            core_module_adoption=core_adoption,
            onboarding_completion=onboarding,
            weekly_logins=weekly_logins,
            time_to_first_value_days=ttfv,
            seat_utilization=seat_util,
            nps_score=nps
        )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Data status
        st.markdown("### 📊 Data Status")
        if df is not None:
            st.success(f"✅ {len(df)} accounts loaded")
        else:
            st.warning("No data loaded")

        # Footer
        st.markdown("""
        <div style="position: fixed; bottom: 20px; left: 20px; right: 20px; max-width: 260px;">
            <p style="color: #4b5563; font-size: 0.75rem; text-align: center;">
                ⚠️ Data stored in session only<br>
                Not persisted between sessions
            </p>
        </div>
        """, unsafe_allow_html=True)

        return region, benchmarks

# =============================================================================
# CHART COMPONENTS
# =============================================================================

def render_risk_distribution_chart(risk_data: Dict) -> None:
    """Render risk distribution donut chart."""
    categories = risk_data["categories"]

    labels = ["Product Risk", "Process Risk", "Development Risk", "Relationship Risk"]
    values = [
        categories["product"]["percentage"],
        categories["process"]["percentage"],
        categories["development"]["percentage"],
        categories["relationship"]["percentage"]
    ]
    colors = [COLORS["accent_red"], COLORS["accent_orange"], COLORS["accent_yellow"], COLORS["accent_purple"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color='#0e1117', width=2)),
        textinfo='percent',
        textfont=dict(size=12, color='white', family='JetBrains Mono'),
    )])

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='#8b949e', size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=60, l=20, r=20),
        height=300,
        annotations=[dict(text=f'<b>{sum(values):.0f}%</b>', x=0.5, y=0.5, font=dict(size=20, color='white'), showarrow=False)]
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_arr_by_risk_chart(df: pd.DataFrame) -> None:
    """Render ARR by risk tier bar chart."""
    arr_by_tier = df.groupby('risk_tier')['arr_value'].sum().reset_index()
    arr_by_tier.columns = ['Risk Tier', 'ARR']

    fig = px.bar(
        arr_by_tier,
        x='Risk Tier',
        y='ARR',
        color='Risk Tier',
        color_discrete_map={'Low': COLORS['accent_green'], 'Medium': COLORS['accent_orange'], 'High': COLORS['accent_red']}
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='#8b949e'),
        yaxis=dict(color='#8b949e', gridcolor='rgba(255,255,255,0.05)'),
        showlegend=False,
        height=300
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="ChurnGuard - AI-Powered ARR Risk Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # Authentication
    if not check_password():
        return

    # Initialize session state
    if "data_seed" not in st.session_state:
        st.session_state.data_seed = 42

    # Data source selection (in main area first)
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 700; color: #ffffff; margin: 0;">
            🛡️ ChurnGuard
        </h1>
        <p style="color: #8b949e; font-size: 1rem;">AI-Powered ARR Risk & Retention Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.expander("📥 Data Source Configuration", expanded=st.session_state.get("df") is None):
        data_source, df = render_data_source_selector()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_source = data_source

    # Use cached data if available
    df = st.session_state.get("df")
    data_source = st.session_state.get("data_source", "sample")

    if df is None:
        st.info("👆 Please configure a data source above to get started.")
        return

    # Sidebar
    region, benchmarks = render_sidebar(df)

    # Filter by region
    if region != "All Regions":
        df_filtered = df[df["region"] == region].copy()
    else:
        df_filtered = df.copy()

    # Compute risk scores
    df_filtered = compute_risk_scores(df_filtered, benchmarks)
    risk_data = compute_risk_attribution(df_filtered, benchmarks)

    # Navigation-based content
    nav = st.session_state.get("nav_selection", "📊 Overview")

    if nav == "📊 Overview":
        render_header(risk_data, region, data_source)

        st.markdown("### Risk Attribution by Category")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_risk_category_card("Product Risk", risk_data["categories"]["product"], "🔧")
        with col2:
            render_risk_category_card("Process Risk", risk_data["categories"]["process"], "⚙️")
        with col3:
            render_risk_category_card("Development Risk", risk_data["categories"]["development"], "📚")
        with col4:
            render_risk_category_card("Relationship Risk", risk_data["categories"]["relationship"], "🤝")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns([1, 2])
        with chart_col1:
            st.markdown("#### Risk Distribution")
            render_risk_distribution_chart(risk_data)
        with chart_col2:
            st.markdown("#### ARR by Risk Tier")
            render_arr_by_risk_chart(df_filtered)

    elif nav == "🤖 AI Insights":
        render_header(risk_data, region, data_source)
        render_ai_insights_section(df_filtered, risk_data)

    elif nav == "📈 Revenue Impact":
        st.markdown("""
        <div style="margin-bottom: 30px;">
            <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ffffff; margin: 0;">
                💰 Revenue Impact Analysis
            </h2>
            <p style="color: #8b949e; margin-top: 8px;">Understand your ARR at risk and potential savings from retention improvements</p>
        </div>
        """, unsafe_allow_html=True)

        total_arr = df_filtered['arr_value'].sum()
        high_risk_arr = df_filtered[df_filtered['risk_tier'] == 'High']['arr_value'].sum()
        med_risk_arr = df_filtered[df_filtered['risk_tier'] == 'Medium']['arr_value'].sum()
        low_risk_arr = df_filtered[df_filtered['risk_tier'] == 'Low']['arr_value'].sum()
        
        # Premium metric cards with gradient backgrounds
        st.markdown("""
        <style>
        .revenue-card {
            background: linear-gradient(135deg, rgba(30,35,45,0.9) 0%, rgba(20,25,35,0.95) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .revenue-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99,102,241,0.4);
            box-shadow: 0 12px 40px rgba(99,102,241,0.15);
        }
        .revenue-card .value {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 12px 0 8px 0;
        }
        .revenue-card .label {
            color: #8b949e;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .revenue-card .delta {
            font-size: 0.85rem;
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 8px;
        }
        .delta-danger { background: rgba(239,68,68,0.2); color: #ef4444; }
        .delta-warning { background: rgba(245,158,11,0.2); color: #f59e0b; }
        .delta-success { background: rgba(34,197,94,0.2); color: #22c55e; }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="revenue-card">
                <div class="label">Total Portfolio ARR</div>
                <div class="value" style="color: #ffffff;">{format_currency(total_arr)}</div>
                <div class="delta delta-success">📊 {len(df_filtered)} accounts</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pct = high_risk_arr/total_arr*100 if total_arr > 0 else 0
            st.markdown(f"""
            <div class="revenue-card">
                <div class="label">🔴 Critical Risk ARR</div>
                <div class="value" style="color: #ef4444;">{format_currency(high_risk_arr)}</div>
                <div class="delta delta-danger">⚠️ {pct:.1f}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            pct = med_risk_arr/total_arr*100 if total_arr > 0 else 0
            st.markdown(f"""
            <div class="revenue-card">
                <div class="label">🟡 Watch List ARR</div>
                <div class="value" style="color: #f59e0b;">{format_currency(med_risk_arr)}</div>
                <div class="delta delta-warning">👀 {pct:.1f}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            pct = low_risk_arr/total_arr*100 if total_arr > 0 else 0
            st.markdown(f"""
            <div class="revenue-card">
                <div class="label">🟢 Healthy ARR</div>
                <div class="value" style="color: #22c55e;">{format_currency(low_risk_arr)}</div>
                <div class="delta delta-success">✅ {pct:.1f}% secure</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        # Two column layout for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### 📊 ARR Waterfall Analysis")
            # Waterfall chart
            waterfall_data = [
                {"label": "Total ARR", "value": total_arr, "color": "#6366f1"},
                {"label": "High Risk", "value": -high_risk_arr, "color": "#ef4444"},
                {"label": "Medium Risk", "value": -med_risk_arr, "color": "#f59e0b"},
                {"label": "Secure ARR", "value": low_risk_arr, "color": "#22c55e"}
            ]
            
            fig = go.Figure(go.Waterfall(
                name="ARR Flow",
                orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Total ARR", "High Risk", "Medium Risk", "Secure ARR"],
                y=[total_arr, -high_risk_arr, -med_risk_arr, low_risk_arr],
                connector={"line": {"color": "rgba(255,255,255,0.2)"}},
                decreasing={"marker": {"color": "#ef4444"}},
                increasing={"marker": {"color": "#22c55e"}},
                totals={"marker": {"color": "#6366f1"}}
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b949e'),
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with chart_col2:
            st.markdown("#### 🎯 Risk Distribution by Industry")
            if 'industry' in df_filtered.columns:
                industry_risk = df_filtered.groupby(['industry', 'risk_tier'])['arr_value'].sum().reset_index()
                # Convert to string to avoid categorical issues
                industry_risk['industry'] = industry_risk['industry'].astype(str)
                industry_risk['risk_tier'] = industry_risk['risk_tier'].astype(str)
                
                fig = px.bar(
                    industry_risk, 
                    x='industry', 
                    y='arr_value',
                    color='risk_tier',
                    color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'},
                    barmode='stack'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    margin=dict(t=20, b=20, l=20, r=20),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=''),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='ARR'),
                    font=dict(color='#8b949e'),
                    legend=dict(title='Risk Tier', font=dict(color='#8b949e'))
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                render_arr_by_risk_chart(df_filtered)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Projected Savings Calculator
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%); 
                    border: 1px solid rgba(99,102,241,0.3); border-radius: 16px; padding: 24px; margin: 20px 0;">
            <h4 style="color: #ffffff; margin-bottom: 16px;">💡 Retention Impact Calculator</h4>
        </div>
        """, unsafe_allow_html=True)
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        with calc_col1:
            retention_improvement = st.slider("Retention Improvement %", 1, 25, 10)
        with calc_col2:
            at_risk_arr = high_risk_arr + med_risk_arr
            potential_save = at_risk_arr * (retention_improvement / 100)
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="color: #8b949e; font-size: 0.9rem;">Potential ARR Saved</div>
                <div style="font-size: 2rem; font-weight: 700; color: #22c55e;">{format_currency(potential_save)}</div>
            </div>
            """, unsafe_allow_html=True)
        with calc_col3:
            # Assuming avg customer lifetime
            lifetime_value = potential_save * 3  # 3-year LTV
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="color: #8b949e; font-size: 0.9rem;">3-Year LTV Impact</div>
                <div style="font-size: 2rem; font-weight: 700; color: #6366f1;">{format_currency(lifetime_value)}</div>
            </div>
            """, unsafe_allow_html=True)

        # Regional breakdown with better visualization
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        region_col1, region_col2 = st.columns(2)
        
        with region_col1:
            st.markdown("#### 🌍 ARR by Region")
            regional_arr = df_filtered.groupby('region').agg({
                'arr_value': 'sum',
                'account_id': 'count',
                'churn_probability': 'mean'
            }).reset_index()
            regional_arr.columns = ['Region', 'ARR', 'Accounts', 'Avg Risk']
            
            fig = px.bar(
                regional_arr, x='Region', y='ARR',
                color='Avg Risk',
                color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444'],
                text=regional_arr['ARR'].apply(format_currency)
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                font=dict(color='#8b949e'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=''),
                coloraxis_showscale=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with region_col2:
            st.markdown("#### 📈 Company Size Distribution")
            if 'company_size' in df_filtered.columns:
                size_arr = df_filtered.groupby('company_size')['arr_value'].sum().reset_index()
                fig = px.pie(size_arr, values='arr_value', names='company_size', hole=0.5)
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(colors=['#6366f1', '#8b5cf6', '#a78bfa'])
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                regional_arr = df_filtered.groupby('region')['arr_value'].sum().reset_index()
                fig = px.pie(regional_arr, values='arr_value', names='region', hole=0.5)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig, use_container_width=True)

    elif nav == "🎯 Benchmarking":
        st.markdown("""
        <div style="margin-bottom: 30px;">
            <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ffffff; margin: 0;">
                🎯 Performance Benchmarking
            </h2>
            <p style="color: #8b949e; margin-top: 8px;">Compare your portfolio performance against industry best practices</p>
        </div>
        """, unsafe_allow_html=True)

        categories_list = ['Core Adoption', 'Onboarding', 'Seat Utilization', 'Training', 'NPS', 'CSM Engagement']
        actual_values = [
            df_filtered['core_module_adoption'].mean() * 100,
            df_filtered['onboarding_completion_pct'].mean() * 100,
            df_filtered['seat_utilization_pct'].mean() * 100,
            df_filtered['training_completion_pct'].mean() * 100,
            df_filtered['nps_score'].mean() * 10,
            df_filtered['csm_engagement_score'].mean() * 100
        ]
        benchmark_values = [
            benchmarks.core_module_adoption * 100,
            benchmarks.onboarding_completion * 100,
            benchmarks.seat_utilization * 100,
            80,
            benchmarks.nps_score * 10,
            80
        ]

        # Premium benchmark comparison cards
        st.markdown("""
        <style>
        .benchmark-card {
            background: linear-gradient(135deg, rgba(30,35,45,0.9) 0%, rgba(20,25,35,0.95) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            margin-bottom: 16px;
        }
        .benchmark-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99,102,241,0.4);
        }
        .benchmark-card .metric-name {
            color: #8b949e;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .benchmark-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .benchmark-card .benchmark-target {
            font-size: 0.85rem;
            color: #6b7280;
        }
        .benchmark-card .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            margin-top: 8px;
        }
        .status-above { background: rgba(34,197,94,0.2); color: #22c55e; }
        .status-below { background: rgba(239,68,68,0.2); color: #ef4444; }
        .status-close { background: rgba(245,158,11,0.2); color: #f59e0b; }
        </style>
        """, unsafe_allow_html=True)

        # Display benchmark comparison cards
        cols = st.columns(3)
        icons = ['🎯', '🚀', '💺', '📚', '⭐', '🤝']
        
        for i, (category, actual, benchmark) in enumerate(zip(categories_list, actual_values, benchmark_values)):
            with cols[i % 3]:
                diff = actual - benchmark
                if diff >= 5:
                    status_class = "status-above"
                    status_text = f"↑ {diff:.1f}% above"
                elif diff <= -5:
                    status_class = "status-below"
                    status_text = f"↓ {abs(diff):.1f}% below"
                else:
                    status_class = "status-close"
                    status_text = f"≈ On target"
                
                color = "#22c55e" if diff >= 0 else "#ef4444"
                
                st.markdown(f"""
                <div class="benchmark-card">
                    <div class="metric-name">{icons[i]} {category}</div>
                    <div class="metric-value" style="color: {color};">{actual:.1f}%</div>
                    <div class="benchmark-target">Target: {benchmark:.0f}%</div>
                    <div class="status-badge {status_class}">{status_text}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Two column layout: Radar chart + Gauge charts
        chart_col1, chart_col2 = st.columns([1, 1])
        
        with chart_col1:
            st.markdown("#### 📊 Performance Radar")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=actual_values + [actual_values[0]],  # Close the polygon
                theta=categories_list + [categories_list[0]],
                fill='toself',
                name='Your Portfolio',
                line_color='#6366f1',
                fillcolor='rgba(99,102,241,0.3)'
            ))
            fig.add_trace(go.Scatterpolar(
                r=benchmark_values + [benchmark_values[0]],
                theta=categories_list + [categories_list[0]],
                fill='toself',
                name='Industry Benchmark',
                line_color='#22c55e',
                fillcolor='rgba(34,197,94,0.15)'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='#6b7280')
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='#8b949e')
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#8b949e')
                ),
                height=450,
                margin=dict(t=30, b=50, l=60, r=60)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with chart_col2:
            st.markdown("#### 🎯 Key Metric Gauges")
            
            # Create gauge charts for top 3 metrics
            gauge_metrics = [
                ("Core Adoption", actual_values[0], benchmark_values[0]),
                ("Onboarding", actual_values[1], benchmark_values[1]),
                ("NPS Score", actual_values[4], benchmark_values[4])
            ]
            
            for metric_name, value, target in gauge_metrics:
                color = "#22c55e" if value >= target else "#ef4444" if value < target - 10 else "#f59e0b"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    delta={'reference': target, 'relative': False, 'position': "bottom"},
                    title={'text': metric_name, 'font': {'size': 14, 'color': '#8b949e'}},
                    number={'font': {'size': 28, 'color': color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2d3139"},
                        'bar': {'color': color},
                        'bgcolor': "rgba(30,35,45,0.5)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, target - 10], 'color': 'rgba(239,68,68,0.2)'},
                            {'range': [target - 10, target], 'color': 'rgba(245,158,11,0.2)'},
                            {'range': [target, 100], 'color': 'rgba(34,197,94,0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "#ffffff", 'width': 2},
                            'thickness': 0.8,
                            'value': target
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=140,
                    margin=dict(t=30, b=10, l=30, r=30)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Improvement recommendations section
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%); 
                    border: 1px solid rgba(99,102,241,0.3); border-radius: 16px; padding: 24px; margin: 20px 0;">
            <h4 style="color: #ffffff; margin-bottom: 20px;">📈 Improvement Opportunities</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Find areas below benchmark
        improvements = []
        for i, (category, actual, benchmark) in enumerate(zip(categories_list, actual_values, benchmark_values)):
            if actual < benchmark:
                gap = benchmark - actual
                improvements.append({
                    'category': category,
                    'actual': actual,
                    'benchmark': benchmark,
                    'gap': gap,
                    'icon': icons[i]
                })
        
        if improvements:
            improvements.sort(key=lambda x: x['gap'], reverse=True)
            
            for imp in improvements[:3]:
                potential_impact = imp['gap'] * 0.1  # Simplified impact calculation
                st.markdown(f"""
                <div style="background: rgba(30,35,45,0.6); border: 1px solid rgba(255,255,255,0.1); 
                            border-radius: 12px; padding: 16px; margin-bottom: 12px;
                            display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <span style="font-size: 1.5rem;">{imp['icon']}</span>
                        <div>
                            <div style="font-weight: 600; color: #ffffff;">{imp['category']}</div>
                            <div style="color: #8b949e; font-size: 0.85rem;">Gap: {imp['gap']:.1f}% below benchmark</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #ef4444; font-weight: 600;">{imp['actual']:.1f}% → {imp['benchmark']:.0f}%</div>
                        <div style="color: #22c55e; font-size: 0.85rem;">+{potential_impact:.1f}% retention lift</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 Congratulations! All metrics are at or above benchmark!")

        # Trend comparison (simulated)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📈 Performance Trend (Last 6 Months)")
        
        # Simulate trend data
        months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        trend_data = []
        for i, month in enumerate(months):
            base_multiplier = 0.85 + (i * 0.03)  # Gradual improvement
            for j, category in enumerate(categories_list):
                trend_data.append({
                    'Month': month,
                    'Category': category,
                    'Value': actual_values[j] * base_multiplier
                })
        
        import pandas as pd
        trend_df = pd.DataFrame(trend_data)
        
        fig = px.line(
            trend_df, x='Month', y='Value', color='Category',
            color_discrete_sequence=['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#22c55e', '#f59e0b']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            font=dict(color='#8b949e'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Score (%)'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    elif nav == "📋 Accounts":
        st.markdown("### 📋 Account Explorer")

        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.multiselect("Risk Tier", ["High", "Medium", "Low"], default=["High", "Medium"])
        with col2:
            if 'industry' in df_filtered.columns:
                industry_filter = st.multiselect("Industry", df_filtered['industry'].unique().tolist())
            else:
                industry_filter = []
        with col3:
            search = st.text_input("Search by company name", "")

        display_df = df_filtered.copy()
        if risk_filter:
            display_df = display_df[display_df['risk_tier'].isin(risk_filter)]
        if industry_filter:
            display_df = display_df[display_df['industry'].isin(industry_filter)]
        if search:
            display_df = display_df[display_df['company_name'].str.contains(search, case=False, na=False)]

        st.markdown(f"**Showing {len(display_df)} accounts**")

        display_cols = ['account_id', 'company_name', 'region', 'arr_value', 'churn_probability', 'risk_tier']
        available_cols = [c for c in display_cols if c in display_df.columns]

        show_df = display_df[available_cols].copy()
        if 'arr_value' in show_df.columns:
            show_df['arr_value'] = show_df['arr_value'].apply(format_currency)
        if 'churn_probability' in show_df.columns:
            show_df['churn_probability'] = (show_df['churn_probability'] * 100).round(0).astype(int).astype(str) + '%'

        st.dataframe(show_df, use_container_width=True, hide_index=True, height=400)

        csv = display_df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "accounts.csv", "text/csv")

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0; border-top: 1px solid #2d3139; margin-top: 40px;">
        <p style="color: #4b5563; font-size: 0.85rem;">
            ChurnGuard v2.0 • AI-Powered by Claude<br>
            <span style="font-size: 0.75rem;">Salesforce • Amplitude • Zendesk Integration Ready</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


