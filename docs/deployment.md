# Deployment Guide: PrecisionCustomer

This document provides deployment guidelines for containerized and cloud environments.

---

## 🐳 Docker Deployment

### 1. Build and Run Image
Ensure Docker is installed on your server, then build the image:
```bash
docker build -t precision-customer:latest .
```

Start the container, passing your configurations as environment variables (or setting up local fallback directories):
```bash
docker run -d -p 5000:5000 \
  -e MONGODB_URL="your-mongodb-connection-string" \
  -e AWS_ACCESS_KEY_ID="your-aws-key" \
  -e AWS_SECRET_ACCESS_KEY="your-aws-secret-key" \
  --name precision-customer-app \
  precision-customer:latest
```

---

## 📦 Docker Compose (Recommended for Multi-Container Testing)

Create a `docker-compose.yml` file in the root directory:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MONGO_DB_URL=${MONGODB_URL}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    restart: always
```

Run compose to spin up:
```bash
docker-compose up -d --build
```

---

## ☁️ Cloud Deployment (AWS ECS / EC2)

### Deployment to AWS EC2
1. Launch an EC2 Instance (Ubuntu 22.04 LTS, t2.micro or higher).
2. Install Docker:
   ```bash
   sudo apt update
   sudo apt install docker.io -y
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   ```
3. Clone this repository and run using the `Docker Deployment` commands listed above.
4. Open port `5000` in the Security Group inbound rules of your EC2 instance.

### GitHub Actions CI/CD Pipeline
PrecisionCustomer includes automatic linting, testing, and Docker builds on every push to the `main` branch.
See `.github/workflows/ci.yml` to view automated steps.
