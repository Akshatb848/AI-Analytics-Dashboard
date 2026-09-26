"""Custom CSS for the dashboard (injected once per page run)."""

CUSTOM_CSS = """
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --secondary-light: #34d399;
        --accent: #f59e0b;
        --accent-light: #fbbf24;
        --danger: #ef4444;
        --warning: #f97316;
        --info: #0ea5e9;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --surface-lighter: #475569;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --text-dim: #64748b;
        --border: rgba(99, 102, 241, 0.2);
        --border-hover: rgba(99, 102, 241, 0.5);
        --glow: rgba(99, 102, 241, 0.4);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 30%, #0f172a 70%, #1e1b4b 100%);
        font-family: 'DM Sans', 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.1) 50%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #10b981 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Version badge */
    .version-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 10px;
        vertical-align: middle;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.7));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #10b981);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.25);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-delta-positive {
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-delta-negative {
        color: #ef4444;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Narrative card */
    .narrative-card {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    
    .narrative-card h4 {
        color: #10b981;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .narrative-card p {
        color: #e2e8f0;
        font-size: 1rem;
    }
    
    .narrative-highlight {
        background: rgba(99, 102, 241, 0.2);
        padding: 2px 6px;
        border-radius: 4px;
        color: #818cf8;
        font-weight: 600;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.08), rgba(16, 185, 129, 0.04));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .insight-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateX(4px);
    }
    
    .insight-card.high-priority {
        border-left: 4px solid #ef4444;
    }
    
    .insight-card.medium-priority {
        border-left: 4px solid #f59e0b;
    }
    
    .insight-card.low-priority {
        border-left: 4px solid #10b981;
    }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
    }
    
    .insight-title {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .insight-description {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    
    .insight-narrative {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        line-height: 1.6;
        border-left: 3px solid #6366f1;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        border-color: rgba(16, 185, 129, 0.6);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
    }
    
    .recommendation-title {
        color: #10b981;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .recommendation-description {
        color: #d1d5db;
        font-size: 0.95rem;
        margin-top: 0.75rem;
        line-height: 1.6;
    }
    
    .recommendation-action {
        color: #34d399;
        font-size: 0.85rem;
        margin-top: 0.75rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 8px;
        display: inline-block;
    }
    
    /* Query container */
    .query-container {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .query-container::before {
        content: '🤖 AI Query Engine';
        position: absolute;
        top: -12px;
        left: 20px;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        color: white;
        font-weight: 600;
    }
    
    /* Tutorial/Help card */
    .tutorial-card {
        background: linear-gradient(145deg, rgba(14, 165, 233, 0.1), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .tutorial-card h4 {
        color: #0ea5e9;
        margin-bottom: 1rem;
    }
    
    .tutorial-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
    }
    
    .tutorial-step-number {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .tutorial-step-content {
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    
    /* Data quality indicator */
    .data-quality {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.7));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .quality-score {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
    }
    
    .quality-score.excellent { color: #10b981; }
    .quality-score.good { color: #f59e0b; }
    .quality-score.poor { color: #ef4444; }
    
    /* Filter tags */
    .filter-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #818cf8;
    }
    
    .filter-tag-remove {
        cursor: pointer;
        color: #ef4444;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background: rgba(99, 102, 241, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #10b981, #f59e0b) !important;
        border-radius: 10px !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .animate-slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        margin-top: 3rem;
        background: linear-gradient(180deg, transparent, rgba(99, 102, 241, 0.05));
    }
    
    /* Suggested queries */
    .suggested-query {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px;
        font-size: 0.85rem;
        color: #818cf8;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .suggested-query:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-1px);
    }
    
    /* Dataset tabs */
    .dataset-tab {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .dataset-tab.active {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        border-color: #6366f1;
        color: white;
    }
    
    .dataset-tab:hover:not(.active) {
        border-color: rgba(99, 102, 241, 0.5);
    }
</style>
"""
