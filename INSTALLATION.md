# Installation Guide

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd googlerag
```

### 2. Backend Setup

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env.sample .env
```

Edit the `.env` file and add your API keys:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Add other required environment variables as needed
```

#### Start Backend Server
```bash
# Using uvicorn directly
uvicorn api.main:app --reload

# Or using Python
python -m uvicorn api.main:app --reload
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

#### Install Dependencies
```bash
cd frontend
npm install --force
```

#### Start Frontend Development Server
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Alternative: Docker Setup

### Using Docker Compose (Recommended)
```bash
# For GPU support
docker-compose -f docker-compose-gpu.yml up --build

# For CPU-only deployment
docker-compose -f docker-compose.yml up --build
```

### Manual Docker Build
```bash
# Build the image
docker build -t agentos .

# Run the container
docker run -p 8000:8000 -p 3000:3000 --env-file .env agentos
```

## Testing the System

### Test Contracts
The system includes test contracts in `/test_contracts/` for testing the Guardian Score analysis:
- `problematic_rental_agreement.txt` - Contains rental exploitation patterns
- `shady_contract.txt` - Contains employment contract violations

Upload these through the web interface to see the Guardian Score system in action.

### API Endpoints
- Health check: `http://localhost:8000/api/v1/health`
- Upload file: `POST http://localhost:8000/api/v1/ragsys/upload-file`
- Query documents: `POST http://localhost:8000/api/v1/ragsys/ask`

## Troubleshooting

### Common Issues

#### Backend Issues
- **Port 8000 already in use**: Kill existing processes or change port in uvicorn command
- **Missing dependencies**: Run `pip install -r requirements.txt` again
- **API key errors**: Ensure `GOOGLE_API_KEY` is set in `.env` file

#### Frontend Issues
- **Node modules errors**: Try `npm install --force` or delete `node_modules` and reinstall
- **Port 3000 in use**: Next.js will automatically use the next available port
- **Build errors**: Ensure Node.js version is 16+

#### Docker Issues
- **Permission errors**: Ensure Docker daemon is running
- **Build failures**: Try `docker system prune` to clean up

### Logs and Debugging
- Backend logs: Check terminal where uvicorn is running
- Frontend logs: Check browser developer console
- API logs: Available in the backend terminal output

## Development Notes

### File Structure
```
AgentOS/
├── api/                 # Backend FastAPI application
├── frontend/           # Next.js frontend application
├── test_contracts/     # Sample malicious contracts for testing
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Docker configuration
└── .env               # Environment variables
```

### Key Features
- **Document Upload**: PDF, DOCX, TXT file support
- **Guardian Score**: AI-powered contract analysis
- **Chat Interface**: Query documents with natural language
- **Contract Classification**: Automatic contract type detection

For detailed API documentation, visit `http://localhost:8000/docs` when the backend is running.