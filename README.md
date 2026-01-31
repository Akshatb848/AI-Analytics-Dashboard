# 🚀 AI Analytics Dashboard

**An Open-Source Alternative to Tableau AI**

A comprehensive analytics platform featuring automated insights, predictive analytics, and natural language queries—powered by Prophet, Plotly, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
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
- Ask questions in plain English
- Automatic query interpretation
- Dynamic visualization generation
- Support for aggregations, filters, and comparisons
- Example queries for guidance

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
   sdk_version: 1.28.0
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
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # Documentation
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
