# Docker Setup for AI Stock Prediction System

This document explains how to run the AI Stock Prediction System using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)
- NVIDIA Docker runtime (if using GPU acceleration)
- At least 8GB RAM available for Docker
- 20GB+ free disk space

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and database passwords
# REQUIRED: Update FMP_API_KEY and ALPHA_VANTAGE_API_KEY
nano .env
```

### 2. Build and Run

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f stock-prediction-app

# Check service status
docker-compose ps
```

### 3. Verify Installation

```bash
# Check if the application is running
curl http://localhost:5000/health

# Check database connection
docker-compose exec stock-prediction-app conda run -n pytorch python -c "import mysql.connector; print('MySQL connection test')"
```

## Services

### Core Services

- **stock-prediction-app**: Main AI application (Port 5000, 8000)
- **mysql**: MySQL database (Port 3306)
- **redis**: Redis cache (Port 6379)

### Optional Services

- **stock-frontend**: React frontend (Port 3000)
- **nginx**: Reverse proxy (Port 80, 443)

## Configuration

### Database

The MySQL service automatically creates the `TradeMine` database. Update connection settings in `.env`:

```env
MYSQL_ROOT_PASSWORD=your_secure_password
MYSQL_DATABASE=TradeMine
MYSQL_USER=stockuser  
MYSQL_PASSWORD=your_user_password
```

### API Keys

**REQUIRED**: Add your API keys to `.env`:

```env
FMP_API_KEY=your_fmp_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

### Model Configuration

```env
ENABLE_MINI_RETRAIN=1        # Enable mini-retraining every 3 days
SAVE_MODEL_AFTER_ONLINE=1    # Save models after online learning
```

## GPU Support

For NVIDIA GPU acceleration:

```bash
# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU support
docker-compose --profile gpu up --build -d
```

## Data Persistence

Volumes are automatically created for:

- `mysql_data`: Database files
- `redis_data`: Cache data
- `./models`: Trained models
- `./data`: Stock data
- `./logs`: Application logs

## Commands

### Development

```bash
# View application logs
docker-compose logs -f stock-prediction-app

# Access application shell
docker-compose exec stock-prediction-app conda run -n pytorch bash

# Run specific model training
docker-compose exec stock-prediction-app conda run -n pytorch python LSTM_model/LSTM_model.py

# Run predictions
docker-compose exec stock-prediction-app conda run -n pytorch python Preproces/Autotrainmodel.py
```

### Maintenance

```bash
# Stop services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Update images
docker-compose pull
docker-compose up --build -d

# View resource usage
docker stats
```

### Backup

```bash
# Backup database
docker-compose exec mysql mysqldump -u root -p TradeMine > backup.sql

# Backup models
docker cp stock_prediction_app:/app/LSTM_model/best_hypertuned_model.keras ./backup/
docker cp stock_prediction_app:/app/GRU_Model/best_hypertuned_model.keras ./backup/
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Increase Docker memory limit to 8GB+ in Docker Desktop settings
   # Or use docker-compose.override.yml to limit service memory
   ```

2. **Database Connection Failed**
   ```bash
   # Check if MySQL is running
   docker-compose ps mysql
   
   # View MySQL logs
   docker-compose logs mysql
   
   # Test connection
   docker-compose exec mysql mysql -u root -p -e "SHOW DATABASES;"
   ```

3. **Model Loading Errors**
   ```bash
   # Check if model files exist
   docker-compose exec stock-prediction-app ls -la /app/LSTM_model/
   
   # Retrain models if corrupted
   docker-compose exec stock-prediction-app conda run -n pytorch python LSTM_model/LSTM_model.py
   ```

4. **API Key Issues**
   ```bash
   # Verify environment variables
   docker-compose exec stock-prediction-app env | grep API
   
   # Update .env and restart
   docker-compose down
   docker-compose up -d
   ```

### Performance Tuning

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  stock-prediction-app:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

## Security Notes

- Never commit `.env` files with real API keys
- Use strong passwords for database
- Consider using Docker secrets for production
- Regularly update base images for security patches

## Production Deployment

For production use:

1. Use external managed database (AWS RDS, Google Cloud SQL)
2. Use container orchestration (Kubernetes, Docker Swarm)
3. Implement proper logging and monitoring
4. Use HTTPS with valid certificates
5. Set up automated backups
6. Configure health checks and auto-restart policies