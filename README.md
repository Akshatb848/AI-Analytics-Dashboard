# 🚀 AI Analytics Dashboard

**An Open-Source Alternative to Tableau AI**

A comprehensive analytics platform featuring automated insights, predictive analytics, and natural language queries—powered by Prophet, Plotly, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 📊 **Automated Data Insights**
- Statistical analysis with anomaly detection
- Trend identification and pattern recognition
- Correlation discovery between variables
- Distribution analysis with skewness detection
- Priority-based insight categorization

### 📈 **Predictive Analytics**
- Time series forecasting using Facebook Prophet
- Configurable seasonality (yearly, weekly, daily)
- Confidence interval visualization
- Trend and component decomposition
- Exportable forecast data

### 💬 **Natural Language Queries**
- Ask questions in plain English, answered with a chart, table and short explanation
- Totals, averages, counts, top/bottom N, one or two groupings ("sales by region and channel")
- Filters by category value, number or year ("average profit in North where discount >= 0.2 in 2025")
- Trends by day, week, month, quarter or year, correlations, distributions and outliers
- Optional **GLM-4.5-Flash** (Z.ai, free tier) to interpret free-form questions — see below

### 🧠 **Semantic Catalog (Governed Metrics)**
- Upload a JSON semantic catalog to define metrics, dimensions, and time grains
- Standardize KPIs across dashboards with consistent definitions
- Validation feedback for missing columns and mappings

### 🗂️ **Executive Dashboard Studio**
- Save AI insights and charts as reusable dashboard cards
- Curate stakeholder-ready summaries from natural language analysis
- Manage and remove cards directly within the app

### 📋 **Data Explorer**
- Interactive data preview
- Column statistics and summaries
- Export capabilities (CSV, JSON)
- Automatic data type detection

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Visualization | Plotly |
| Forecasting | Prophet |
| Data Processing | Pandas, NumPy |
| Statistics | SciPy |

---

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-analytics-dashboard.git
cd ai-analytics-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Dependencies in `requirements.txt` are pinned to the versions the app is tested against, which require Python 3.11 or newer.

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite in `tests/` runs the app headlessly with Streamlit's `AppTest` (sample data, Ask Data queries, forecasting, Data Tools, Dashboards) and unit-tests the data cleaning helpers.

### Using Google Colab

```python
# Install dependencies
!pip install streamlit pandas numpy plotly prophet scipy pyngrok

# Write the app file
%%writefile app.py
# ... (paste the full app.py content)

# Run with ngrok tunnel
from pyngrok import ngrok
!streamlit run app.py &>/dev/null &
public_url = ngrok.connect(8501)
print(f"Access your app at: {public_url}")
```

---

## ☁️ Deployment Options

### Docker (any host)

```bash
docker build -t ai-analytics-dashboard .
docker run -d -p 8501:8501 \
  -e ZAI_API_KEY="your-key" \
  ai-analytics-dashboard
```

The app is served on port 8501 and runs as a non-root user. The image has a health check on
Streamlit's `/_stcore/health` endpoint (`docker inspect -f '{{.State.Health.Status}}' <container>`),
which load balancers and orchestrators can also poll. `ZAI_API_KEY` is optional; secrets files are
excluded from the image by `.dockerignore`, so pass keys at runtime.

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/ai-analytics-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

3. **Your app will be live at:**
   ```
   https://yourusername-ai-analytics-dashboard.streamlit.app
   ```

### Option 2: Railway (Free Tier Available)

1. **Create `Procfile`:**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create `railway.json`:**
   ```json
   {
     "$schema": "https://railway.app/railway.schema.json",
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
     }
   }
   ```

3. **Deploy:**
   - Go to [railway.app](https://railway.app)
   - Connect GitHub repository
   - Railway auto-detects and deploys

### Option 3: Render (Free Tier)

1. **Create `render.yaml`:**
   ```yaml
   services:
     - type: web
       name: ai-analytics-dashboard
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy:**
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect repository

### Option 4: Hugging Face Spaces (Free)

1. **Create `README.md` for HF:**
   ```yaml
   ---
   title: AI Analytics Dashboard
   emoji: 📊
   colorFrom: indigo
   colorTo: green
   sdk: streamlit
   sdk_version: 1.64.0
   app_file: app.py
   pinned: false
   ---
   ```

2. **Deploy:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Create new Space with Streamlit SDK
   - Upload files or connect GitHub

---

## 📁 Project Structure

```
ai-analytics-dashboard/
├── app.py                 # Streamlit page: layout, session state, caching
├── analytics/
│   ├── data_utils.py      # Type detection, cleaning, quality score, sample data
│   ├── preprocessing.py   # Data Tools operations (missing values, outliers, date features)
│   ├── insights.py        # Automated insights and narratives
│   ├── query_engine.py    # Ask Data question answering
│   ├── forecasting.py     # Prophet forecasts
│   ├── visualization.py   # Overview charts
│   ├── reports.py         # Markdown / HTML / CSV reports
│   └── formatting.py      # HTML escaping and number formatting
├── ui/
│   ├── styles.py          # Custom CSS
│   └── components.py      # Tutorial and small UI helpers
├── semantic_engine.py     # Column profiling and semantic catalog
├── requirements.txt       # Python dependencies (pinned)
├── requirements-dev.txt   # Test dependencies
├── tests/                 # Smoke and unit tests
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # Documentation
├── Dockerfile            # Production image with health check
├── Procfile              # For Railway/Heroku
└── .gitignore            # Git ignore file
```

---

## 🎯 Usage Guide

### Uploading Data

1. Click "Upload CSV" in the sidebar
2. Select your CSV file
3. The app automatically detects:
   - Date columns for time series
   - Numeric columns for analysis
   - Categorical columns for grouping

### Supported Data Formats

| Column Type | Detection Method | Use Case |
|-------------|-----------------|----------|
| Date | Auto-parse datetime | Time series, trends |
| Numeric | Float/Int types | Metrics, aggregations |
| Categorical | String/Object types | Grouping, filtering |

### Natural Language Query Examples

```
# Aggregations
"Total sales by region"
"Average profit by category"
"Maximum quantity by month"

# Time Series
"Trend of sales over time"
"Show revenue growth"

# Comparisons
"Compare sales and profit"
"Correlation between price and quantity"

# Filtering
"Sales where region is North"
"Top 5 products by revenue"
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STREAMLIT_SERVER_PORT` | Server port | 8501 |
| `STREAMLIT_SERVER_HEADLESS` | Headless mode | true |
| `MAX_UPLOAD_ROWS` | Maximum rows read from an uploaded file; larger files are truncated with a warning | 200000 |

Uploads are limited to 50 MB via `server.maxUploadSize` in `.streamlit/config.toml`.

### GLM-4.5-Flash for Ask Data (optional)

Without an API key, Ask Data uses the built-in rule-based parser. With a Z.ai key, questions
are interpreted by **GLM-4.5-Flash** (free, rate-limited), which handles freer phrasing.

1. Create an API key at [z.ai](https://z.ai) (API keys page of the Z.ai open platform).
2. Provide it as `ZAI_API_KEY`, either as an environment variable:

   ```bash
   export ZAI_API_KEY="your-key"
   streamlit run app.py
   ```

   or in `.streamlit/secrets.toml` (already in `.gitignore`; on Streamlit Cloud use the app's
   **Secrets** settings instead):

   ```toml
   ZAI_API_KEY = "your-key"
   ```

| Variable | Description | Default |
|----------|-------------|---------|
| `ZAI_API_KEY` | Z.ai API key; enables GLM in Ask Data | not set |
| `ZAI_MODEL` | Model name | `glm-4.5-flash` |
| `ZAI_BASE_URL` | API base URL. Keys from the GLM Coding Plan use `https://api.z.ai/api/coding/paas/v4` | `https://api.z.ai/api/paas/v4` |

**How it works and what is sent.** The model only turns the question into a query plan (which
columns, filters, grouping, aggregation); the app validates that plan against your columns and
runs it with pandas, so the model never executes code. Z.ai receives the question plus column
names, column types, up to 15 values per category column and the date range — **never the data
rows**. If the key is missing, the API is unreachable or rate-limited, or the reply can't be used,
the built-in parser answers and the app says so. A toggle in the Ask Data tab turns GLM off, and
"How this was answered" shows the plan that ran.

### Custom Theming

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#6366f1"      # Indigo
backgroundColor = "#0f172a"    # Dark slate
secondaryBackgroundColor = "#1e293b"
textColor = "#f1f5f9"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - feel free to use for personal and commercial projects.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) - App framework
- [Prophet](https://facebook.github.io/prophet/) - Time series forecasting
- [Plotly](https://plotly.com) - Interactive visualizations

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/ai-analytics-dashboard/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/ai-analytics-dashboard/discussions)

---

<p align="center">
  <strong>Built with ❤️ as an open-source alternative to enterprise analytics tools</strong>
</p>
