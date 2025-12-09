# Tracker - Vehicle Detection and Tracking System

A comprehensive vehicle tracking system with Python backend for AI-powered detection and Next.js frontend for management and visualization.

## Features

- **Real-time Vehicle Detection**: Process video feeds to detect and track vehicles
- **AI Classification**: Automatic truck body type classification using xAI Grok VLM
- **Job Management**: Track processing jobs with progress monitoring
- **Web Dashboard**: Modern Next.js interface for uploading videos and viewing results
- **REST API**: FastAPI backend with comprehensive endpoints
- **Containerized**: Docker support for easy deployment

## Architecture

- **Backend**: Python 3.11 + FastAPI + YOLO + SAM2 + xAI Grok
- **Frontend**: Next.js 16 + TypeScript + React 19 + Tailwind CSS
- **Database**: SQLite (development) / PostgreSQL (production)
- **Deployment**: Docker + Docker Compose, AWS ECS/Fargate ready

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Plutonotfromspace/Tracker.git
   cd Tracker
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and set your XAI_API_KEY
   ```

3. **Start the application**:
   ```bash
   docker compose up -d
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

For detailed Docker deployment instructions, see [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md).

### Manual Setup

#### Backend

1. **Install Python dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Download YOLO models**:
   ```bash
   # Download models to backend/models/
   # e.g., yolov8n.pt, yolov8s.pt, etc.
   ```

3. **Set environment variables**:
   ```bash
   export XAI_API_KEY=your_api_key
   ```

4. **Start the server**:
   ```bash
   uvicorn server:app --reload
   ```

#### Frontend

1. **Install Node.js dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Set environment variables**:
   ```bash
   export NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

## Usage

### Processing a Video

**Via Command Line**:
```bash
cd backend
python main.py --video path/to/video.mp4 --job-id auto
```

**Via Web Interface**:
1. Navigate to http://localhost:3000/upload
2. Upload a video file
3. Monitor progress on the jobs page
4. View results and classified trucks

### API Usage

The backend provides a REST API:

```bash
# Get all jobs
curl http://localhost:8000/api/jobs/

# Get job details
curl http://localhost:8000/api/jobs/{job_id}

# Get trucks for a job
curl http://localhost:8000/api/trucks/job/{job_id}
```

Full API documentation: http://localhost:8000/docs

## Project Structure

```
Tracker/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # CLI entry point
│   ├── server.py           # FastAPI server
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # YOLO model files
│   ├── src/                # Source code
│   │   └── tracker/        # Main package
│   │       ├── api/        # API endpoints
│   │       ├── classification/  # AI classification
│   │       ├── detection/  # Object detection
│   │       ├── jobs/       # Job management
│   │       └── data/       # Database models
│   ├── tools/              # Utility scripts
│   └── Dockerfile          # Backend container
├── frontend/               # Next.js TypeScript frontend
│   ├── app/                # Next.js app directory
│   ├── components/         # React components
│   ├── package.json        # Node dependencies
│   └── Dockerfile          # Frontend container
├── docker-compose.yml      # Local development setup
├── docker-compose.prod.yml # Production overrides
├── .env.example            # Environment template
└── DOCKER_DEPLOYMENT.md    # Deployment guide
```

## Configuration

### Backend Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `XAI_API_KEY` | xAI Grok API key for classification | Yes |
| `DATABASE_URL` | Database connection string | No (defaults to SQLite) |
| `CORS_ORIGINS` | Allowed CORS origins | No |

### Frontend Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | Yes |

## Development

### Backend Development

```bash
cd backend
# Install dev dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest

# Run linter
pylint src/
```

### Frontend Development

```bash
cd frontend
# Install dependencies
npm install

# Run development server
npm run dev

# Run linter
npm run lint

# Build for production
npm run build
```

## Deployment

### Docker Deployment

See [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) for comprehensive deployment instructions.

### AWS ECS/Fargate

1. Build and push Docker images to ECR
2. Create ECS task definitions
3. Deploy services with proper networking and secrets
4. Configure Application Load Balancer
5. Set up auto-scaling and monitoring

Detailed steps in [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md).

## Features in Detail

### Vehicle Detection
- YOLO-based object detection
- Multi-model support (YOLOv8n/s/m/x)
- Frame-by-frame tracking
- Zone-based entry/exit detection

### Classification
- xAI Grok Vision Language Model integration
- Automatic truck body type classification
- Confidence scoring
- Batch processing support

### Job Management
- Asynchronous job processing
- Progress tracking with stages
- Database persistence
- Web-based monitoring

### Web Dashboard
- Modern, responsive UI
- Real-time progress updates
- Image gallery with filtering
- Export capabilities (JSON/CSV)

## Requirements

### Backend
- Python 3.11+
- OpenCV
- PyTorch (for YOLO/SAM2)
- FastAPI
- SQLModel

### Frontend
- Node.js 20+
- Next.js 16
- React 19
- TypeScript

### System
- 4GB+ RAM
- GPU recommended for processing
- Docker (for containerized deployment)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues or questions:
1. Check the [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) for deployment issues
2. Review API documentation at `/docs`
3. Open an issue on GitHub

## Credits

Built with:
- [YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [SAM2](https://github.com/facebookresearch/segment-anything-2) for segmentation
- [xAI Grok](https://x.ai/) for classification
- [FastAPI](https://fastapi.tiangolo.com/) for backend API
- [Next.js](https://nextjs.org/) for frontend
