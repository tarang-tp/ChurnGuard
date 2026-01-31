# ChurnGuard Deployment Guide

Complete guide to deploying ChurnGuard as a public website with custom domain support.

---

## Table of Contents

1. [Quick Start (Local Development)](#1-quick-start-local-development)
2. [Streamlit Community Cloud (Free, Fastest)](#2-streamlit-community-cloud)
3. [Render.com (Custom Domain, Free Tier)](#3-rendercom-deployment)
4. [AWS Deployment (Production-Grade)](#4-aws-deployment)
5. [Docker Deployment](#5-docker-deployment)
6. [Security Considerations](#6-security-considerations)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Quick Start (Local Development)

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone or create project directory
mkdir churnguard && cd churnguard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys

# Run the application
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## 2. Streamlit Community Cloud

**Best for**: Quick demos, prototypes, sharing with team  
**Limitations**: Subdomain only (*.streamlit.app), limited resources

### Step-by-Step

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial ChurnGuard deployment"
gh repo create churnguard --private --source=. --push
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select repository: `your-username/churnguard`
   - Branch: `main`
   - Main file: `app.py`

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to app settings → Secrets
   - Add:
   ```toml
   APP_PASSWORD = "your-secure-password"
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   ```

4. **Deploy** → App live at: `https://your-app-name.streamlit.app`

---

## 3. Render.com Deployment

**Best for**: Custom domains, free tier, easy SSL  
**Cost**: Free (with sleep), $7/mo always-on

### Step-by-Step

1. **Create `render.yaml`** in project root:
```yaml
services:
  - type: web
    name: churnguard
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: APP_PASSWORD
        sync: false
```

2. **Push to GitHub**
```bash
git add render.yaml
git commit -m "Add Render configuration"
git push
```

3. **Deploy on Render**
   - Sign up at [render.com](https://render.com)
   - New → Web Service → Connect GitHub repo
   - Render auto-detects `render.yaml`
   - Click "Create Web Service"

4. **Configure Environment Variables**
   - Dashboard → Your Service → Environment
   - Add: `ANTHROPIC_API_KEY`, `APP_PASSWORD`

5. **Custom Domain Setup**
   - Dashboard → Your Service → Settings → Custom Domains
   - Add your domain: `churnguard.yourcompany.com`
   - Add DNS records as instructed:
   ```
   Type: CNAME
   Name: churnguard
   Value: your-service.onrender.com
   ```
   - Render provides free SSL automatically

---

## 4. AWS Deployment

**Best for**: Production, scalability, enterprise requirements  
**Cost**: ~$10-50/mo depending on usage

### Option A: AWS App Runner (Easiest)

1. **Create `Dockerfile`**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Push to ECR**:
```bash
# Build and tag
docker build -t churnguard .

# Create ECR repo
aws ecr create-repository --repository-name churnguard

# Login and push
aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker tag churnguard:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/churnguard:latest
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/churnguard:latest
```

3. **Deploy via App Runner**:
   - AWS Console → App Runner → Create Service
   - Source: Container registry → ECR
   - Configure: 1 vCPU, 2GB RAM, Port 8501
   - Environment variables: Add your secrets
   - Deploy

4. **Custom Domain**:
   - App Runner → Your Service → Custom Domains
   - Add domain, configure DNS with provided records
   - SSL auto-provisioned

### Option B: EC2 with Nginx (Full Control)

1. **Launch EC2 Instance**:
   - Ubuntu 22.04 LTS, t3.small or larger
   - Security group: Allow 80, 443, 22

2. **Setup Server**:
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update and install
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv nginx certbot python3-certbot-nginx -y

# Clone and setup app
git clone https://github.com/your-username/churnguard.git
cd churnguard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create secrets
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
APP_PASSWORD = "your-password"
ANTHROPIC_API_KEY = "sk-ant-..."
EOF
```

3. **Create Systemd Service**:
```bash
sudo cat > /etc/systemd/system/churnguard.service << EOF
[Unit]
Description=ChurnGuard Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/churnguard
Environment="PATH=/home/ubuntu/churnguard/venv/bin"
ExecStart=/home/ubuntu/churnguard/venv/bin/streamlit run app.py --server.port 8501 --server.address 127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable churnguard
sudo systemctl start churnguard
```

4. **Configure Nginx**:
```bash
sudo cat > /etc/nginx/sites-available/churnguard << EOF
server {
    listen 80;
    server_name churnguard.yourcompany.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/churnguard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

5. **Setup SSL with Certbot**:
```bash
sudo certbot --nginx -d churnguard.yourcompany.com
```

6. **DNS Configuration**:
   - Add A record: `churnguard` → Your EC2 Elastic IP

---

## 5. Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  churnguard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - APP_PASSWORD=${APP_PASSWORD}
    volumes:
      - ./.streamlit:/app/.streamlit:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Run with Docker

```bash
# Build
docker build -t churnguard .

# Run
docker run -d \
  -p 8501:8501 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e APP_PASSWORD="your-password" \
  --name churnguard \
  churnguard

# Or with docker-compose
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "APP_PASSWORD=your-password" >> .env
docker-compose up -d
```

---

## 6. Security Considerations

### For Production Deployment

1. **Authentication**
   - Replace simple password with [Streamlit-Authenticator](https://github.com/mkhorasani/Streamlit-Authenticator)
   - Or integrate Auth0/Okta for SSO:
   ```python
   # pip install streamlit-authenticator
   import streamlit_authenticator as stauth
   
   authenticator = stauth.Authenticate(
       credentials, cookie_name, key, cookie_expiry_days
   )
   ```

2. **HTTPS**
   - Always use HTTPS in production
   - Render/App Runner: Auto-provisioned
   - EC2: Use Certbot/Let's Encrypt
   - Never expose HTTP on public internet

3. **API Key Security**
   - Never commit secrets to git
   - Use environment variables or secret managers
   - Rotate keys periodically
   - Consider AWS Secrets Manager or HashiCorp Vault

4. **Data Protection**
   - Data is in-memory only (by design)
   - For persistent storage, add encryption at rest
   - Consider GDPR/SOC2 requirements
   - Implement data retention policies

5. **Network Security**
   - Use VPC for AWS deployments
   - Restrict IP ranges if possible
   - Enable WAF for DDoS protection

### Compliance Checklist

- [ ] HTTPS enforced
- [ ] Authentication implemented
- [ ] Secrets not in code
- [ ] Audit logging enabled
- [ ] Data encryption at rest
- [ ] Privacy policy published
- [ ] GDPR data export/deletion capability

---

## 7. Troubleshooting

### Common Issues

**App won't start**
```bash
# Check logs
streamlit run app.py 2>&1 | tee app.log

# Verify dependencies
pip list | grep streamlit
pip list | grep anthropic
```

**Port already in use**
```bash
# Find and kill process
lsof -i :8501
kill -9 <PID>
```

**Secrets not loading**
```bash
# Verify secrets file exists
cat .streamlit/secrets.toml

# Check permissions
chmod 600 .streamlit/secrets.toml
```

**API connection errors**
- Verify API keys are correct
- Check network connectivity
- Review API rate limits
- Enable debug logging

**Memory issues on free tier**
- Reduce data sample size
- Disable unused features
- Upgrade to paid tier

### Getting Help

- Streamlit Docs: https://docs.streamlit.io
- Anthropic Docs: https://docs.anthropic.com
- GitHub Issues: Create issue in your repo
- Streamlit Community: https://discuss.streamlit.io

---

## Quick Reference

| Platform | Custom Domain | Free Tier | Setup Time | Best For |
|----------|--------------|-----------|------------|----------|
| Streamlit Cloud | ❌ | ✅ | 5 min | Demos |
| Render | ✅ | ✅ (sleep) | 15 min | Startups |
| AWS App Runner | ✅ | ❌ | 30 min | Scale |
| AWS EC2 | ✅ | ❌ | 1 hour | Enterprise |
| Docker | ✅ | N/A | 10 min | Self-host |

---

**Need help?** Open an issue or reach out to the team.
