"""
Flask API Server for Vehicle Damage Assessment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
from dotenv import load_dotenv
from pipeline import DamageAssessmentPipeline
from chat_agent import AutoShopChatAgent
import requests

# Try to import Pipecat agent, but make it optional
try:
    from pipecat_agent import PipecatAutoShopAgent
    PIPECAT_AVAILABLE = True
except (ImportError, SyntaxError, TypeError) as e:
    print(f"⚠ Pipecat agent not available: {e}")
    print("  Using standard chat agent only. Pipecat requires Python 3.10+.")
    PipecatAutoShopAgent = None
    PIPECAT_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize pipeline (lazy loading)
pipeline = None

def get_pipeline():
    """Lazy load the pipeline to avoid loading model on startup."""
    global pipeline
    if pipeline is None:
        # Check for model file
        model_path = os.getenv('MODEL_PATH', 'runs/detect/car_damage_yolov8/weights/best.pt')
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = [
                'best.pt',
                'yolov8n.pt',  # Fallback to pretrained model
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
        
        pipeline = DamageAssessmentPipeline(
            model_path=model_path,
            cost_table_path='part_costs.csv',
            ml_model_path=None,
            hourly_rate=120.0,  # Industry standard: $100-150/hr
            ml_weight=0.0
        )
    return pipeline

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/api/assess', methods=['POST'])
def assess_damage():
    """
    Assess vehicle damage from uploaded image.
    
    Expects:
    - 'image': base64 encoded image or file upload
    - 'vehicle_info' (optional): JSON with make, model, year, mileage
    """
    try:
        # Get image data
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        elif 'image' in request.json:
            # Base64 encoded image
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert PIL image to OpenCV format
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save temporary image for pipeline
        temp_path = 'temp_upload.jpg'
        cv2.imwrite(temp_path, image_rgb)
        
        # Get vehicle info (optional)
        vehicle_info = request.json.get('vehicle_info', {}) if request.is_json else {}
        if not vehicle_info and 'vehicle_info' in request.form:
            import json
            vehicle_info = json.loads(request.form['vehicle_info'])
        
        # Get confidence threshold
        conf_threshold = float(request.json.get('conf_threshold', 0.5)) if request.is_json else 0.5
        
        # Process image
        pipeline = get_pipeline()
        result = pipeline.process_image(
            image_path=temp_path,
            vehicle_info=vehicle_info,
            conf_threshold=conf_threshold
        )
        
        # Generate visualization with bounding boxes
        vis_image = pipeline.visualize_results(
            image_path=temp_path,
            result=result,
            show=False
        )
        
        # Convert visualization to base64
        _, buffer = cv2.imencode('.jpg', vis_image)
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Get vehicle valuation info from first damage instance (if available)
        vehicle_multiplier = 1.0
        valuation_info = None
        if result['damage_instances'] and 'cost_estimate' in result['damage_instances'][0]:
            first_estimate = result['damage_instances'][0]['cost_estimate']
            vehicle_multiplier = first_estimate.get('vehicle_multiplier', 1.0)
            valuation_info = first_estimate.get('valuation_info', None)
        
        # Format response
        response = {
            'success': True,
            'num_damages': result['num_damages'],
            'total_estimated_cost': round(result['total_estimated_cost'], 2),
            'visualization': f'data:image/jpeg;base64,{vis_base64}',
            'vehicle_multiplier': round(vehicle_multiplier, 3),
            'valuation_info': valuation_info,
            'damage_instances': []
        }
        
        for instance in result['damage_instances']:
            cost_est = instance['cost_estimate']
            response['damage_instances'].append({
                'damage_class': instance['damage_class'],
                'severity_class': instance['severity_class'],
                'severity_score': round(instance['severity_score'], 2),
                'part_name': instance['part_name'],
                'confidence': round(instance['confidence'], 2),
                'cost_estimate': {
                    'final_cost': round(cost_est['final_cost'], 2),
                    'base_cost': round(cost_est.get('base_cost', cost_est['final_cost']), 2),
                    'vehicle_multiplier': round(cost_est.get('vehicle_multiplier', 1.0), 3),
                    'rule_breakdown': {
                        'part_cost': round(cost_est['rule_breakdown'].get('part_cost', 0), 2),
                        'labor_cost': round(cost_est['rule_breakdown'].get('labor_cost', 0), 2),
                        'paint_cost': round(cost_est['rule_breakdown'].get('paint_cost', 0), 2),
                        'shop_supplies': round(cost_est['rule_breakdown'].get('shop_supplies', 0), 2),
                        'disposal_fee': round(cost_est['rule_breakdown'].get('disposal_fee', 0), 2),
                        'additional_fees': round(cost_est['rule_breakdown'].get('additional_fees', 0), 2),
                        'replace_or_repair': cost_est['rule_breakdown'].get('replace_or_repair', 'repair')
                    }
                },
                'bbox': instance['bbox']
            })
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/api/visualize', methods=['POST'])
def visualize_damage():
    """
    Generate visualization of damage assessment.
    
    Expects same input as /api/assess
    Returns base64 encoded image with annotations
    """
    try:
        # Get image data (same as assess endpoint)
        if 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        elif 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert to OpenCV format
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        temp_path = 'temp_upload.jpg'
        cv2.imwrite(temp_path, image_rgb)
        
        # Get vehicle info
        vehicle_info = request.json.get('vehicle_info', {}) if request.is_json else {}
        conf_threshold = float(request.json.get('conf_threshold', 0.25)) if request.is_json else 0.25
        
        # Process and visualize
        pipeline = get_pipeline()
        result = pipeline.process_image(
            image_path=temp_path,
            vehicle_info=vehicle_info,
            conf_threshold=conf_threshold
        )
        
        # Generate visualization
        vis_image = pipeline.visualize_results(
            image_path=temp_path,
            result=result,
            show=False
        )
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', vis_image)
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'visualization': f'data:image/jpeg;base64,{vis_base64}'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Initialize chat agents (lazy loading)
chat_agent = None
pipecat_agent = None

def get_chat_agent():
    """Lazy load the chat agent."""
    global chat_agent
    if chat_agent is None:
        chat_agent = AutoShopChatAgent()
        # Check if Groq is working
        if chat_agent.client:
            print("✓ Groq API is configured and ready")
        else:
            print("⚠ Groq API not configured - using fallback responses")
    return chat_agent

def get_pipecat_agent():
    """Lazy load the Pipecat agent."""
    global pipecat_agent
    if not PIPECAT_AVAILABLE or PipecatAutoShopAgent is None:
        return None
    if pipecat_agent is None:
        try:
            pipecat_agent = PipecatAutoShopAgent()
        except Exception as e:
            print(f"⚠ Error initializing Pipecat agent: {e}")
            return None
    return pipecat_agent

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Groq's Whisper API."""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty audio file'
            }), 400
        
        # Get Groq API key
        api_key = os.getenv('GROQ_API_KEY', '').strip()
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'GROQ_API_KEY not found'
            }), 500
        
        from groq import Groq
        groq_client = Groq(api_key=api_key)
        
        # Read audio file
        audio_bytes = audio_file.read()
        
        # Create a file-like object for Groq API
        # Groq's Whisper API expects a tuple: (filename, file_object, content_type)
        audio_io = io.BytesIO(audio_bytes)
        filename = audio_file.filename or 'recording.webm'
        content_type = audio_file.content_type or 'audio/webm'
        
        # Transcribe using Whisper Large v3 Turbo (fast and accurate)
        # Groq's Whisper API is compatible with OpenAI's format
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, audio_io, content_type),
            model='whisper-large-v3-turbo',
            language='en'
        )
        
        transcript_text = transcription.text.strip()
        
        return jsonify({
            'success': True,
            'transcript': transcript_text
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/api/crop-damage-box', methods=['POST'])
def crop_damage_box():
    """Crop a specific damage box from the original image."""
    try:
        data = request.json
        image_data = data.get('image')  # Base64 encoded image
        bbox = data.get('bbox')  # [x1, y1, x2, y2]
        
        if not image_data or not bbox:
            return jsonify({'error': 'Image and bbox are required'}), 400
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert to OpenCV format
        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array
        
        # Extract bbox coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding (20% on each side)
        h, w = image_cv.shape[:2]
        padding_x = int((x2 - x1) * 0.2)
        padding_y = int((y2 - y1) * 0.2)
        
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)
        
        # Crop the image
        cropped = image_cv[y1:y2, x1:x2]
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'cropped_image': f'data:image/jpeg;base64,{cropped_base64}'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the conversational agent."""
    try:
        data = request.json
        message = data.get('message', '')
        conversation_history = data.get('conversation_history', [])
        damage_results = data.get('damage_results', None)
        use_pipecat = data.get('use_pipecat', False)  # Optional flag to use Pipecat
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Generate response - try Pipecat first if requested, otherwise use standard agent
        box_indices = None
        if use_pipecat and PIPECAT_AVAILABLE:
            try:
                pipecat_agent = get_pipecat_agent()
                if pipecat_agent:
                    response_data = pipecat_agent.process_message(
                        message=message,
                        conversation_history=conversation_history,
                        damage_results=damage_results
                    )
                    # Pipecat might return string or dict
                    if isinstance(response_data, dict):
                        response = response_data.get('response', '')
                        box_indices = response_data.get('box_indices')
                    else:
                        response = response_data
                else:
                    # Pipecat not available, use standard agent
                    agent = get_chat_agent()
                    response_data = agent.generate_response(
                        message=message,
                        conversation_history=conversation_history,
                        damage_results=damage_results
                    )
                    if isinstance(response_data, dict):
                        response = response_data.get('response', '')
                        box_indices = response_data.get('box_indices')
                    else:
                        response = response_data
            except Exception as e:
                print(f"Pipecat agent error, falling back to standard agent: {e}")
                agent = get_chat_agent()
                response_data = agent.generate_response(
                    message=message,
                    conversation_history=conversation_history,
                    damage_results=damage_results
                )
                if isinstance(response_data, dict):
                    response = response_data.get('response', '')
                    box_indices = response_data.get('box_indices')
                else:
                    response = response_data
        else:
            agent = get_chat_agent()
            response_data = agent.generate_response(
                message=message,
                conversation_history=conversation_history,
                damage_results=damage_results
            )
            # Handle both string and dict responses
            if isinstance(response_data, dict):
                response = response_data.get('response', '')
                box_indices = response_data.get('box_indices')
            else:
                response = response_data
        
        result = {
            'success': True,
            'response': response
        }
        
        # Add box indices if detected (can be single or multiple)
        if box_indices is not None:
            result['box_indices'] = box_indices
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))  # Changed to 5001 to avoid macOS AirPlay conflict
    app.run(host='0.0.0.0', port=port, debug=True)

