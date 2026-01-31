# 🛡️ ChurnGuard

**AI-Powered ARR Risk & Retention Dashboard**

A production-grade Streamlit dashboard that integrates Salesforce, Amplitude, and Zendesk data to identify churn risk and provides AI-powered retention recommendations using Claude.

![Dashboard Preview](docs/dashboard-preview.png)

## ✨ Features

### 📊 Data Ingestion
- **Sample Data Mode**: Demo with realistic synthetic customer data
- **CSV Upload**: Import data from exported files
- **API Integration**: Direct connections to:
  - **Salesforce**: Account and ARR data via SOQL
  - **Amplitude**: Product usage and engagement metrics
  - **Zendesk**: Support ticket analysis

### 🎯 Risk Attribution
- **Product Risk**: Module adoption, seat utilization, feature gaps
- **Process Risk**: Onboarding completion, login frequency, TTFV
- **Development Risk**: Training completion, API integrations
- **Relationship Risk**: NPS scores, CSM engagement, support health

### 🤖 AI-Powered Insights
- **Google Gemini Integration**: Deep analysis of merged customer data (primary)
- **Anthropic Claude Support**: Alternative AI provider option
- **Pain Point Detection**: Identifies top 4-6 churn drivers
- **Root Cause Analysis**: Evidence-based explanations
- **Actionable Recommendations**: Prioritized retention strategies with ARR impact estimates

### 🎨 Modern UI
- Dark theme with SaaS-grade aesthetics
- Interactive charts and visualizations
- Responsive card-based layout
- Region and benchmark filtering

## 🚀 Quick Start

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

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys

# Run
streamlit run app.py
```

### Docker

```bash
# Using docker-compose
docker-compose up -d

# Or standalone
docker build -t churnguard .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY="sk-ant-..." churnguard
```

## 📦 Project Structure

```
churnguard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Docker Compose setup
├── render.yaml           # Render.com deployment
├── DEPLOYMENT.md         # Deployment guide
├── .streamlit/
│   ├── config.toml       # Streamlit theme config
│   └── secrets.toml.example  # Secrets template
└── .gitignore
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key for AI insights | Yes (for AI features) |
| `ANTHROPIC_API_KEY` | Alternative: Claude API key | No (optional) |
| `APP_PASSWORD` | Dashboard access password | No (default: churnguard2024) |

### Benchmark Defaults

Configure in sidebar or modify in code:

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

- Simple password authentication (production: use OAuth/SSO)
- Secrets stored in environment variables
- Data processed in-memory only (not persisted)
- HTTPS recommended for all deployments

For production deployments, consider:
- Streamlit-Authenticator or Auth0 integration
- AWS Secrets Manager for credentials
- VPC/firewall restrictions
- GDPR compliance measures

## 🛣️ Roadmap

- [ ] Multi-tenant database (PostgreSQL)
- [ ] Real OAuth flows for integrations
- [ ] Background data sync with Celery
- [ ] ML-based churn prediction model
- [ ] SHAP values for explainability
- [ ] Email/Slack alerting
- [ ] Role-based access control

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

- GitHub Issues: Bug reports and feature requests
- Documentation: [docs/](docs/)

---

Built with ❤️ using Streamlit and Google Gemini AI
