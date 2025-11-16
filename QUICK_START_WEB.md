# Quick Start - Web Application

Get the BadgerBuild web application running in minutes!

## Prerequisites

1. **Python 3.8** with virtual environment set up
2. **Node.js** (v14 or higher) - [Download here](https://nodejs.org/)
3. **Trained YOLOv8 model** at `runs/detect/car_damage_yolov8/weights/best.pt`

## Quick Setup

### Step 1: Install Backend Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Flask and dependencies
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 3: Start the Application

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
python app.py
```
Backend will run on `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Frontend will open at `http://localhost:3000`

### Alternative: Use Startup Scripts

```bash
# Terminal 1
./start_backend.sh

# Terminal 2
./start_frontend.sh
```

## Using the Application

1. Open `http://localhost:3000` in your browser
2. Upload a vehicle image (drag & drop or click to browse)
3. Optionally enter vehicle information
4. Click "Assess Damage"
5. View results with total cost and detailed breakdown

## Troubleshooting

### Backend Issues

- **ModuleNotFoundError**: Make sure virtual environment is activated and dependencies are installed
- **Model not found**: Check that `runs/detect/car_damage_yolov8/weights/best.pt` exists
- **Port 5000 in use**: Change port in `app.py` or set `PORT` environment variable

### Frontend Issues

- **npm not found**: Install Node.js from [nodejs.org](https://nodejs.org/)
- **CORS errors**: Make sure backend is running and Flask-CORS is installed
- **Port 3000 in use**: React will prompt to use a different port

## Features

✅ Modern, responsive UI  
✅ Drag-and-drop image upload  
✅ Real-time damage assessment  
✅ Detailed cost breakdown  
✅ Severity classification  
✅ Part identification  
✅ Mobile-friendly design  

## Next Steps

- Train your own YOLOv8 model (see `START_TRAINING.md`)
- Customize part costs in `part_costs.csv`
- Add ML cost model for improved accuracy
- Deploy to production (Heroku, AWS, etc.)

