# 🛡️ ChurnGuard

**AI-Powered ARR Risk & Retention Dashboard**

A production-grade Streamlit dashboard that identifies customer churn risk and provides AI-powered retention recommendations. Integrates with Salesforce, Amplitude, and Zendesk data sources.

## ✨ Features

### 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | High-level portfolio health, risk distribution, and key metrics |
| **AI Insights** | AI-powered churn analysis with actionable retention strategies |
| **Revenue Impact** | ARR waterfall analysis, retention calculator, and regional breakdown |
| **Benchmarking** | Performance radar, gauge charts, and improvement recommendations |
| **Accounts** | Filterable account explorer with export functionality |

### 🎯 Risk Attribution
- **Product Risk**: Module adoption, seat utilization, feature gaps
- **Process Risk**: Onboarding completion, login frequency, TTFV
- **Development Risk**: Training completion, API integrations
- **Relationship Risk**: NPS scores, CSM engagement, support health

### 🤖 AI-Powered Insights
- **Keywords AI Integration** (Primary): Supports multiple LLM providers through one API
- **Google Gemini Support**: Direct integration with Gemini models
- **Anthropic Claude Support**: Alternative AI provider option
- **Pain Point Detection**: Identifies top 4-6 churn drivers
- **Root Cause Analysis**: Evidence-based explanations
- **Actionable Recommendations**: Prioritized retention strategies with ARR impact estimates

### 📈 Premium Visualizations
- Interactive waterfall charts for ARR analysis
- Radar charts for benchmark comparison
- Animated gauge charts for key metrics
- Stacked bar charts for industry risk breakdown
- Trend analysis with historical performance

### 🎨 Modern UI
- Dark theme with premium SaaS aesthetics
- Gradient cards with hover animations
- Responsive card-based layout
- Region and benchmark filtering

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Local Development

```bash
# Clone repository
git clone https://github.com/your-username/churnguard.git
cd churnguard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets (optional - can also input keys in UI)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys

# Run
streamlit run app.py
```

### Docker

```bash
# Build and run
docker build -t churnguard .
docker run -p 8501:8501 churnguard

# Or using docker-compose
docker-compose up -d
```

Then open **http://localhost:8501** in your browser.

## 📦 Project Structure

```
churnguard/
├── app.py                    # Main Streamlit application (~2400 lines)
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Docker Compose setup
├── render.yaml              # Render.com deployment
├── DEPLOYMENT.md            # Deployment guide
├── .streamlit/
│   ├── config.toml          # Streamlit theme config
│   └── secrets.toml.example # Secrets template
└── .gitignore
```

## 🔧 Configuration

### API Keys

You can configure API keys in two ways:

1. **In the UI**: Enter your API key directly on the AI Insights page
2. **In secrets.toml**: Add keys to `.streamlit/secrets.toml`

| Provider | Key Name | Get Your Key |
|----------|----------|--------------|
| Keywords AI | `KEYWORDS_API_KEY` | [keywordsai.co](https://keywordsai.co) |
| Google Gemini | `GOOGLE_API_KEY` | [Google AI Studio](https://makersuite.google.com/app/apikey) |
| Anthropic Claude | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |

### Benchmark Defaults

Configure in the sidebar or modify in code:

| Metric | Default | Description |
|--------|---------|-------------|
| Core Module Adoption | 80% | Target adoption rate |
| Onboarding Completion | 90% | Target completion rate |
| Weekly Logins | 5 | Expected logins per week |
| Time to First Value | 14 days | Max acceptable TTFV |
| Seat Utilization | 75% | Target utilization |
| NPS Score | 8.0 | Target NPS |

## 🌐 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:

- Streamlit Community Cloud (free, fastest)
- Render.com (custom domain, free tier)
- AWS App Runner / EC2 (production-grade)
- Docker self-hosting

### Quick Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 🔒 Security

- Secrets stored in environment variables (not hardcoded)
- Data processed in-memory only (not persisted)
- HTTPS recommended for all deployments
- `.streamlit/secrets.toml` is gitignored

For production deployments, consider:
- Streamlit-Authenticator or Auth0 integration
- AWS Secrets Manager for credentials
- VPC/firewall restrictions
- GDPR compliance measures

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **AI Providers**: Keywords AI, Google Gemini, Anthropic Claude
- **Containerization**: Docker

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

- GitHub Issues: Bug reports and feature requests

---

Built with ❤️ using Streamlit and TRAE AI
