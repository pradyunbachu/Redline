# BadgerBuild Frontend - Vehicle Damage Assessment Web App

A sleek React frontend for the vehicle damage assessment system.

## Setup Instructions

### 1. Install Backend Dependencies

Make sure you're in the virtual environment and install Flask dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

Navigate to the frontend directory and install Node.js dependencies:

```bash
cd frontend
npm install
```

**Note:** You need Node.js (v14 or higher) installed. If you don't have it, download from [nodejs.org](https://nodejs.org/).

### 3. Start the Backend Server

In one terminal, activate the virtual environment and start Flask:

```bash
source .venv/bin/activate
python app.py
```

The backend will run on `http://localhost:5000`

### 4. Start the Frontend Development Server

In another terminal, navigate to the frontend directory and start React:

```bash
cd frontend
npm start
```

The frontend will open automatically in your browser at `http://localhost:3000`

## Usage

1. Open the web application in your browser
2. Click the upload area or drag and drop a vehicle image
3. Optionally fill in vehicle information (make, model, year, mileage)
4. Click "Assess Damage"
5. View the results showing:
   - Total estimated repair cost
   - Number of damage instances
   - Detailed breakdown for each damage (type, severity, part, cost breakdown)

## Features

- **Modern UI**: Clean, gradient-based design with smooth animations
- **Image Upload**: Drag-and-drop or click to upload
- **Real-time Processing**: See loading state while analyzing
- **Detailed Results**: 
   - Damage type and severity classification
   - Affected part identification
   - Cost breakdown (parts, labor, paint)
   - Confidence scores
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## API Endpoints

The backend provides two main endpoints:

- `POST /api/assess` - Assess damage from uploaded image
- `POST /api/visualize` - Get annotated visualization (not yet integrated in frontend)
- `GET /api/health` - Health check

## Configuration

To change the API URL (if backend runs on different port), create a `.env` file in the `frontend` directory:

```
REACT_APP_API_URL=http://localhost:5000
```

## Production Build

To build the frontend for production:

```bash
cd frontend
npm run build
```

This creates an optimized build in the `frontend/build` directory that can be served by any static file server.

## Troubleshooting

- **CORS errors**: Make sure Flask-CORS is installed and the backend is running
- **Model not found**: Ensure you have a trained YOLOv8 model at `runs/detect/car_damage_yolov8/weights/best.pt` or update the `MODEL_PATH` in `app.py`
- **Port already in use**: Change the port in `app.py` (backend) or set `PORT` environment variable

