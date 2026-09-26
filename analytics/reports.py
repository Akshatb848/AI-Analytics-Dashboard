"""Markdown, HTML and CSV report generation."""
from datetime import datetime
from typing import Dict, List

import pandas as pd

from analytics.data_utils import (
    detect_date_column,
)
from analytics.formatting import _markdown_bold_to_html, safe_html


class ReportGenerator:
    """Generate exportable reports in multiple formats."""
    
    def __init__(self, df: pd.DataFrame, insights: List[Dict], recommendations: List[Dict]):
        self.df = df
        self.insights = insights
        self.recommendations = recommendations
        self.date_col = detect_date_column(df)
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary markdown."""
        high_priority = [i for i in self.insights if i.get('priority') == 'high']
        
        date_range = ""
        if self.date_col:
            date_range = f"**Analysis Period:** {self.df[self.date_col].min().strftime('%B %d, %Y')} to {self.df[self.date_col].max().strftime('%B %d, %Y')}"
        
        summary = f"""# 📊 AI Analytics Executive Report
*Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*

---

## Overview

- **Total Records Analyzed:** {len(self.df):,}
- **Data Columns:** {len(self.df.columns)}
- **Critical Findings:** {len(high_priority)}
- {date_range}

---

## 🔴 Key Findings

"""
        for i, insight in enumerate(high_priority[:5], 1):
            summary += f"### {i}. {insight['title']}\n\n"
            summary += f"{insight.get('narrative', insight['description'])}\n\n"
        
        summary += "\n---\n\n## 💡 Recommendations\n\n"
        
        for rec in self.recommendations[:5]:
            summary += f"### {rec['icon']} {rec['title']}\n\n"
            summary += f"{rec['description']}\n\n"
            summary += f"**Action:** {rec['action']}\n\n"
        
        return summary
    
    def generate_html_report(self) -> str:
        """Generate full HTML report."""
        high_priority = [i for i in self.insights if i.get('priority') == 'high']
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Analytics Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #0f172a, #1e1b4b); 
            color: #f1f5f9; 
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ 
            font-size: 2.5rem; 
            background: linear-gradient(135deg, #6366f1, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        h2 {{ color: #818cf8; margin: 30px 0 15px; border-bottom: 2px solid #6366f1; padding-bottom: 10px; }}
        h3 {{ color: #10b981; margin: 20px 0 10px; }}
        .meta {{ color: #94a3b8; margin-bottom: 30px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ 
            background: rgba(30, 41, 59, 0.8); 
            padding: 20px; 
            border-radius: 12px; 
            border: 1px solid rgba(99, 102, 241, 0.3);
            flex: 1;
            min-width: 150px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2rem; color: #6366f1; font-weight: bold; }}
        .stat-label {{ color: #94a3b8; font-size: 0.9rem; }}
        .insight {{ 
            background: rgba(99, 102, 241, 0.1); 
            border-left: 4px solid #6366f1;
            padding: 20px; 
            border-radius: 8px; 
            margin: 15px 0;
        }}
        .insight-high {{ border-left-color: #ef4444; }}
        .insight-medium {{ border-left-color: #f59e0b; }}
        .recommendation {{ 
            background: rgba(16, 185, 129, 0.1); 
            border-left: 4px solid #10b981;
            padding: 20px; 
            border-radius: 8px; 
            margin: 15px 0;
        }}
        .action {{ 
            background: rgba(16, 185, 129, 0.2); 
            padding: 8px 15px; 
            border-radius: 6px; 
            display: inline-block;
            margin-top: 10px;
            color: #34d399;
        }}
        p {{ margin: 10px 0; color: #e2e8f0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 AI Analytics Report</h1>
        <p class="meta">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(self.df):,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.df.columns)}</div>
                <div class="stat-label">Data Columns</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(high_priority)}</div>
                <div class="stat-label">Critical Findings</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.insights)}</div>
                <div class="stat-label">Total Insights</div>
            </div>
        </div>
        
        <h2>🔍 Key Insights</h2>
"""
        
        for insight in self.insights[:10]:
            priority_class = f"insight-{safe_html(insight.get('priority', 'medium'))}"
            html += f"""
        <div class="insight {priority_class}">
            <h3>{safe_html(insight['icon'])} {safe_html(insight['title'])}</h3>
            <p>{_markdown_bold_to_html(safe_html(insight.get('narrative', insight['description'])))}</p>
        </div>
"""
        
        html += "\n        <h2>💡 Recommendations</h2>\n"
        
        for rec in self.recommendations[:5]:
            html += f"""
        <div class="recommendation">
            <h3>{safe_html(rec['icon'])} {safe_html(rec['title'])}</h3>
            <p>{safe_html(rec['description'])}</p>
            <div class="action">→ {safe_html(rec['action'])}</div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>"""
        
        return html
    
    def generate_csv_summary(self) -> str:
        """Generate insights as CSV."""
        rows = []
        for insight in self.insights:
            rows.append({
                'Type': insight['type'],
                'Priority': insight.get('priority', 'medium'),
                'Title': insight['title'],
                'Description': insight['description'],
                'Metric': insight.get('metric', ''),
                'Value': insight.get('value', '')
            })
        return pd.DataFrame(rows).to_csv(index=False)
