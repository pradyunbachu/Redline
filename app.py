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
from pipeline import DamageAssessmentPipeline

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

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))  # Changed to 5001 to avoid macOS AirPlay conflict
    app.run(host='0.0.0.0', port=port, debug=True)

