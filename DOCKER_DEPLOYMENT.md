# Docker Deployment Guide

This guide explains how to containerize and deploy the Tracker application using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose V2
- AWS CLI (for AWS deployment)
- At least 4GB of available RAM

## Quick Start - Local Development

1. **Copy environment variables**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file** and set your environment variables:
   ```bash
   XAI_API_KEY=your_actual_api_key
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Start all services**:
   ```bash
   docker compose up -d
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

5. **View logs**:
   ```bash
   docker compose logs -f
   ```

6. **Stop services**:
   ```bash
   docker compose down
   ```

## Building Individual Services

### Backend

```bash
cd backend
docker build -t tracker-backend .
docker run -p 8000:8000 \
  -e XAI_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  tracker-backend
```

### Frontend

```bash
cd frontend
docker build -t tracker-frontend .
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  tracker-frontend
```

## Production Deployment

### Local Production Testing

Test the production configuration locally:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### AWS ECS/Fargate Deployment

#### 1. Create ECR Repositories

```bash
aws ecr create-repository --repository-name tracker-backend
aws ecr create-repository --repository-name tracker-frontend
```

#### 2. Build and Push Images

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build and tag images
docker build -t tracker-backend:latest ./backend
docker tag tracker-backend:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tracker-backend:latest

docker build -t tracker-frontend:latest ./frontend
docker tag tracker-frontend:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tracker-frontend:latest

# Push images
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tracker-backend:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tracker-frontend:latest
```

#### 3. Create AWS Secrets

Store sensitive data in AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name tracker/xai-api-key \
  --secret-string "your_xai_api_key"

aws secretsmanager create-secret \
  --name tracker/database-url \
  --secret-string "your_database_url"
```

#### 4. Create ECS Task Definition

Create a file `ecs-task-definition.json`:

```json
{
  "family": "tracker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tracker-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "secrets": [
        {
          "name": "XAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:${AWS_ACCOUNT_ID}:secret:tracker/xai-api-key"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "sqlite:///data/tracker.db"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/tracker-backend",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "backend"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    },
    {
      "name": "frontend",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tracker-frontend:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NEXT_PUBLIC_API_URL",
          "value": "http://localhost:8000"
        }
      ],
      "dependsOn": [
        {
          "containerName": "backend",
          "condition": "HEALTHY"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/tracker-frontend",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "frontend"
        }
      }
    }
  ]
}
```

#### 5. Create ECS Service

```bash
aws ecs create-service \
  --cluster your-cluster-name \
  --service-name tracker \
  --task-definition tracker \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

## Volumes and Persistence

### Development

Data is stored in Docker volumes:
- `backend_data`: Database and processed data
- `backend_videos`: Uploaded videos
- `backend_models`: AI models

### Production

For production on AWS:

1. **Use Amazon EFS** for shared file storage:
   ```bash
   aws efs create-file-system --tags Key=Name,Value=tracker-storage
   ```

2. **Or use Amazon S3** for video storage and results.

3. **Use RDS** (PostgreSQL/MySQL) instead of SQLite:
   - Update `DATABASE_URL` environment variable
   - Remove SQLite volume mount

## Health Checks

Both services expose health check endpoints:

- Backend: `GET http://localhost:8000/health`
- Frontend: `GET http://localhost:3000/api/health`

These are used by Docker and AWS ECS for health monitoring.

## Environment Variables

### Backend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `XAI_API_KEY` | Yes | - | xAI Grok API key for classification |
| `DATABASE_URL` | No | `sqlite:///data/tracker.db` | Database connection string |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Allowed CORS origins |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### Frontend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` | Backend API URL |
| `NODE_ENV` | No | `production` | Node environment |

## Troubleshooting

### Backend won't start

1. Check logs: `docker compose logs backend`
2. Verify environment variables are set
3. Ensure required models are available

### Frontend can't connect to backend

1. Verify `NEXT_PUBLIC_API_URL` is correct
2. Check network connectivity: `docker compose exec frontend ping backend`
3. Ensure backend health check passes

### Database issues

1. Check volume permissions
2. Verify `DATABASE_URL` format
3. For SQLite, ensure `/app/data` directory exists

### Out of memory

1. Increase Docker memory limit (Docker Desktop settings)
2. Or increase ECS task memory allocation

## Optimization Tips

1. **Multi-stage builds**: Already implemented for frontend
2. **Layer caching**: Build dependencies layer separately
3. **Image size**: Use slim/alpine base images
4. **.dockerignore**: Exclude unnecessary files from context
5. **Resource limits**: Set CPU/memory limits in production

## Security Best Practices

1. **Never commit secrets** to `.env` files
2. **Use AWS Secrets Manager** for production credentials
3. **Run as non-root user** (frontend already implements this)
4. **Keep base images updated** regularly
5. **Scan images for vulnerabilities**: `docker scan tracker-backend`
6. **Use private ECR** repositories

## Monitoring and Logging

### CloudWatch Logs

Logs are automatically sent to CloudWatch when using the production configuration:
- Backend logs: `/ecs/tracker-backend`
- Frontend logs: `/ecs/tracker-frontend`

### Health Checks

Monitor health check status:
```bash
docker compose ps
aws ecs describe-services --cluster your-cluster --services tracker
```

## Backup and Recovery

### Database Backup

```bash
# Backup SQLite database
docker compose exec backend tar czf /tmp/backup.tar.gz /app/data
docker cp tracker-backend:/tmp/backup.tar.gz ./backup.tar.gz
```

### Restore from Backup

```bash
docker cp ./backup.tar.gz tracker-backend:/tmp/
docker compose exec backend tar xzf /tmp/backup.tar.gz -C /
```

## Scaling

For AWS ECS:

```bash
aws ecs update-service \
  --cluster your-cluster-name \
  --service tracker \
  --desired-count 3
```

## Support

For issues or questions:
1. Check logs: `docker compose logs`
2. Verify configuration: `docker compose config`
3. Test health endpoints
4. Review this documentation

## Next Steps

1. Set up CI/CD pipeline for automated builds
2. Configure Application Load Balancer for production
3. Set up auto-scaling policies
4. Implement blue-green deployments
5. Add monitoring and alerting
