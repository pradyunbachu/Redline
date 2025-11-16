"""
Main Pipeline: Damage Detection → Severity → Cost Estimation

End-to-end pipeline that:
1. Runs YOLOv8 detection model on vehicle image (bounding boxes)
2. Extracts severity features from bounding boxes
3. Estimates repair costs using hybrid rule-based + ML approach

FIXED VERSION: Works with detection model (bboxes), not segmentation (masks)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import cv2

from severity_features import SeverityFeatureExtractor
from cost_estimator import HybridCostEstimator


class DamageAssessmentPipeline:
    """
    Complete pipeline for vehicle damage assessment using YOLOv8 detection.
    """
    
    def __init__(
        self,
        model_path: str,
        cost_table_path: str = "part_costs.csv",
        ml_model_path: Optional[str] = None,
        hourly_rate: float = 80.0,
        ml_weight: float = 0.3
    ):
        """
        Args:
            model_path: Path to trained YOLOv8 detection model (.pt file)
            cost_table_path: Path to part cost table CSV
            ml_model_path: Optional path to trained ML cost model
            hourly_rate: Shop hourly labor rate
            ml_weight: Weight for ML cost prediction
        """
        # Load YOLOv8 model
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        
        self.feature_extractor = SeverityFeatureExtractor()
        # Use realistic hourly rate (minimum $100/hr)
        actual_hourly_rate = max(hourly_rate, 100.0) if hourly_rate < 100 else hourly_rate
        self.cost_estimator = HybridCostEstimator(
            cost_table_path=cost_table_path,
            ml_model_path=ml_model_path,
            hourly_rate=actual_hourly_rate,
            ml_weight=ml_weight
        )
    
    def process_image(
        self,
        image_path: str,
        vehicle_bbox: Optional[tuple] = None,
        vehicle_info: Optional[Dict] = None,
        conf_threshold: float = 0.01  # Confidence threshold for detections
    ) -> Dict:
        """
        Process a vehicle image and return damage assessment.
        
        Args:
            image_path: Path to vehicle image
            vehicle_bbox: Optional bounding box of vehicle (x1, y1, x2, y2)
            vehicle_info: Optional vehicle info (make, model, year, mileage)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with damage instances, severity, and cost estimates
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection model
        predictions = self._run_detection_model(image_rgb, conf_threshold)
        
        # If no vehicle bbox provided, use full image
        if vehicle_bbox is None:
            h, w = image_rgb.shape[:2]
            vehicle_bbox = (0, 0, w, h)
        
        # Process each detected damage instance
        damage_instances = []
        total_cost = 0.0
        
        for pred in predictions:
            instance_result = self._process_damage_instance(
                pred, image_rgb, vehicle_bbox, vehicle_info
            )
            damage_instances.append(instance_result)
            total_cost += instance_result['cost_estimate']['final_cost']
        
        return {
            'image_path': image_path,
            'num_damages': len(damage_instances),
            'damage_instances': damage_instances,
            'total_estimated_cost': total_cost,
            'vehicle_info': vehicle_info or {}
        }
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _merge_overlapping_detections(self, predictions: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Merge overlapping detections using IoU to avoid duplicates.
        
        Args:
            predictions: List of detection dictionaries
            iou_threshold: IoU threshold above which boxes are considered duplicates
            
        Returns:
            Filtered list with duplicates merged
        """
        if len(predictions) == 0:
            return []
        
        # Sort by confidence (highest first)
        sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        merged = []
        used = [False] * len(sorted_preds)
        
        for i, pred1 in enumerate(sorted_preds):
            if used[i]:
                continue
            
            bbox1 = pred1['bbox']
            merged_pred = pred1.copy()
            count = 1
            
            # Check for overlapping boxes
            for j, pred2 in enumerate(sorted_preds[i+1:], start=i+1):
                if used[j]:
                    continue
                
                bbox2 = pred2['bbox']
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > iou_threshold:
                    # Merge: keep higher confidence, average bbox, combine classes if same
                    used[j] = True
                    count += 1
                    
                    # Average bbox coordinates (weighted by confidence)
                    w1 = pred1['confidence']
                    w2 = pred2['confidence']
                    total_w = w1 + w2
                    
                    merged_bbox = (
                        (bbox1[0] * w1 + bbox2[0] * w2) / total_w,
                        (bbox1[1] * w1 + bbox2[1] * w2) / total_w,
                        (bbox1[2] * w1 + bbox2[2] * w2) / total_w,
                        (bbox1[3] * w1 + bbox2[3] * w2) / total_w
                    )
                    merged_pred['bbox'] = np.array(merged_bbox)
                    
                    # Use higher confidence
                    merged_pred['confidence'] = max(pred1['confidence'], pred2['confidence'])
            
            merged.append(merged_pred)
            used[i] = True
        
        return merged
    
    def _run_detection_model(
        self, 
        image: np.ndarray,
        conf_threshold: float = 0.01
    ) -> List[Dict]:
        """
        Run YOLOv8 detection model and return predictions.
        Runs detection on the full image normally.
        
        Args:
            image: RGB image (H, W, 3)
            conf_threshold: Confidence threshold
            
        Returns:
            List of dicts with keys:
            - bbox: (x1, y1, x2, y2)
            - class: damage class name
            - confidence: detection confidence
        """
        # Run detection on full image
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        predictions = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(boxes.conf[i])
                
                if confidence >= conf_threshold:
                    class_id = int(boxes.cls[i])
                    class_name = result.names[class_id]
                    
                    predictions.append({
                        'bbox': bbox,
                        'class': class_name,
                        'confidence': confidence,
                        'class_id': class_id
                    })
        
        return predictions
    
    def _process_damage_instance(
        self,
        prediction: Dict,
        image: np.ndarray,
        vehicle_bbox: Optional[tuple],
        vehicle_info: Optional[Dict]
    ) -> Dict:
        """
        Process a single damage instance through the pipeline.
        
        Since we're using detection (bboxes), we create a simple rectangular
        mask from the bounding box for feature extraction.
        """
        bbox = prediction['bbox']
        damage_class = prediction.get('class', 'unknown')
        confidence = prediction.get('confidence', 1.0)
        
        # Create a rectangular mask from bbox (for feature extraction)
        mask = self._bbox_to_mask(bbox, image.shape[:2])
        
        # Extract severity features
        features = self.feature_extractor.extract_features(
            mask=mask,
            image=image,
            bbox=bbox,
            vehicle_bbox=vehicle_bbox,
            confidence=confidence
        )
        
        # Compute severity class
        severity_class, severity_score = self.feature_extractor.compute_severity_class(features)
        
        # Determine affected part from bbox location
        part_name = self._infer_part_from_bbox(bbox, image.shape)
        
        # Estimate cost
        cost_estimate = self.cost_estimator.estimate(
            part_name=part_name,
            damage_class=damage_class,
            severity_class=severity_class,
            features=features,
            vehicle_info=vehicle_info
        )
        
        return {
            'damage_class': damage_class,
            'severity_class': severity_class,
            'severity_score': severity_score,
            'part_name': part_name,
            'features': features,
            'cost_estimate': cost_estimate,
            'bbox': bbox.tolist(),
            'confidence': confidence
        }
    
    def _bbox_to_mask(self, bbox: np.ndarray, image_shape: tuple) -> np.ndarray:
        """
        Convert bounding box to binary mask.
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_shape: (height, width)
            
        Returns:
            Binary mask (H, W)
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Ensure bounds are within image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_shape[1], x2)
        y2 = min(image_shape[0], y2)
        
        mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def _infer_part_from_bbox(self, bbox: np.ndarray, image_shape: tuple) -> str:
        """
        Infer affected part from bounding box location.
        
        Improved heuristic that better detects rear and side parts.
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_shape: (height, width, channels)
            
        Returns:
            Part name string
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        img_h, img_w = image_shape[:2]
        
        # Normalize to 0-1
        norm_x = center_x / img_w
        norm_y = center_y / img_h
        norm_width = bbox_width / img_w
        norm_height = bbox_height / img_h
        
        # More granular detection zones
        # Top quarter (y < 0.25) - Roof, hood, windshield area
        if norm_y < 0.25:
            if norm_x < 0.2:
                return 'headlight'
            elif norm_x > 0.8:
                return 'taillight'
            elif norm_x < 0.4:
                return 'headlight'
            elif norm_x > 0.6:
                return 'taillight'
            else:
                return 'hood'
        
        # Upper-middle (0.25 <= y < 0.45) - Hood, windshield, roof
        elif norm_y < 0.45:
            if norm_x < 0.2:
                return 'front_fender'
            elif norm_x > 0.8:
                return 'rear_fender'
            elif norm_x < 0.4:
                return 'hood'
            elif norm_x > 0.6:
                return 'trunk'
            else:
                return 'hood' if norm_x < 0.5 else 'trunk'
        
        # Middle section (0.45 <= y < 0.65) - Doors, fenders, side panels
        elif norm_y < 0.65:
            if norm_x < 0.2:
                return 'front_fender'
            elif norm_x > 0.8:
                return 'rear_fender'
            elif norm_x < 0.35:
                return 'front_door'
            elif norm_x > 0.65:
                return 'rear_door'
            elif norm_x < 0.5:
                return 'door'
            else:
                return 'door'
        
        # Lower-middle (0.65 <= y < 0.85) - Doors, bumpers, side panels
        elif norm_y < 0.85:
            if norm_x < 0.2:
                return 'front_fender'
            elif norm_x > 0.8:
                return 'rear_fender'
            elif norm_x < 0.35:
                return 'front_door'
            elif norm_x > 0.65:
                return 'rear_door'
            elif norm_x < 0.5:
                return 'door'
            else:
                return 'door'
        
        # Bottom quarter (y >= 0.85) - Bumpers, wheels, lower panels
        else:
            if norm_x < 0.2:
                return 'front_bumper'
            elif norm_x > 0.8:
                return 'rear_bumper'
            elif norm_x < 0.35:
                return 'front_bumper'
            elif norm_x > 0.65:
                return 'rear_bumper'
            else:
                # Use aspect ratio to distinguish
                if norm_width > norm_height * 1.5:
                    # Wide bbox likely bumper
                    return 'bumper' if norm_x < 0.5 else 'rear_bumper'
                else:
                    return 'bumper'
    
    def process_batch(
        self,
        image_paths: List[str],
        vehicle_info: Optional[Dict] = None,
        conf_threshold: float = 0.01
    ) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            vehicle_info: Optional vehicle info (applied to all images)
            conf_threshold: Confidence threshold
            
        Returns:
            List of results dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.process_image(
                    image_path=image_path,
                    vehicle_info=vehicle_info,
                    conf_threshold=conf_threshold
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'num_damages': 0,
                    'damage_instances': [],
                    'total_estimated_cost': 0.0
                })
        
        return results
    
    def visualize_results(
        self,
        image_path: str,
        result: Dict,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image_path: Path to original image
            result: Result dictionary from process_image()
            output_path: Optional path to save visualization
            show: Whether to display image
            
        Returns:
            Annotated image (BGR format for cv2)
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes and labels
        for i, damage in enumerate(result['damage_instances']):
            bbox = damage['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on severity
            severity = damage['severity_class']
            if severity == 'minor':
                color = (0, 255, 0)  # Green
            elif severity == 'moderate':
                color = (0, 165, 255)  # Orange
            else:  # severe
                color = (0, 0, 255)  # Red
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            damage_class = damage['damage_class']
            cost = damage['cost_estimate']['final_cost']
            part = damage['part_name']
            conf = damage['confidence']
            
            label = f"{damage_class} ({severity})"
            label2 = f"{part}: ${cost:.0f} ({conf:.2f})"
            
            # Draw label background
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (w2, h2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(image, (x1, y1 - h1 - h2 - 10), (x1 + max(w1, w2), y1), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x1, y1 - h2 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, label2, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw total cost
        total_cost = result['total_estimated_cost']
        num_damages = result['num_damages']
        summary = f"Total: {num_damages} damages, ${total_cost:.2f}"
        
        cv2.putText(image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Saved visualization to {output_path}")
        
        # Display if requested
        if show:
            cv2.imshow('Damage Assessment', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image


def example_usage():
    """Example of how to use the fixed pipeline."""
    
    # Initialize pipeline with your trained model
    pipeline = DamageAssessmentPipeline(
        model_path='runs/detect/car_damage_yolov8/weights/best.pt',  # Your downloaded YOLOv8 detection model
        cost_table_path='part_costs.csv',
        ml_model_path=None,  # Add when ML model is trained
        hourly_rate=80.0,
        ml_weight=0.0  # Pure rule-based for now
    )
    
    # Process a single image
    result = pipeline.process_image(
        image_path='images/test1.jpg',
        vehicle_info={
            'make': 'Toyota',
            'model': 'Camry',
            'year': 2020,
            'mileage': 30000
        }
    )
    
    # Print results
    print("=" * 60)
    print(f"Damage Assessment Results")
    print("=" * 60)
    print(f"Image: {result['image_path']}")
    print(f"Found {result['num_damages']} damage instance(s)")
    print(f"Total estimated cost: ${result['total_estimated_cost']:.2f}")
    print()
    
    for i, instance in enumerate(result['damage_instances'], 1):
        print(f"Damage #{i}:")
        print(f"  Type: {instance['damage_class']}")
        print(f"  Severity: {instance['severity_class']} (score: {instance['severity_score']:.2f})")
        print(f"  Part: {instance['part_name']}")
        print(f"  Confidence: {instance['confidence']:.2f}")
        print(f"  Estimated Cost: ${instance['cost_estimate']['final_cost']:.2f}")
        print(f"    - Parts: ${instance['cost_estimate']['rule_breakdown']['part_cost']:.2f}")
        print(f"    - Labor: ${instance['cost_estimate']['rule_breakdown']['labor_cost']:.2f}")
        print(f"    - Paint: ${instance['cost_estimate']['rule_breakdown']['paint_cost']:.2f}")
        print(f"    - Action: {instance['cost_estimate']['rule_breakdown']['replace_or_repair']}")
        print()
    
    # Visualize results
    pipeline.visualize_results(
        image_path='images/test1.jpg',
        result=result,
        output_path='result_visualization.jpg',
        show=False
    )
    
    print("Visualization saved to result_visualization.jpg")
    
    # Process multiple images
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60)
    
    image_paths = ['images/test1.jpg']
    batch_results = pipeline.process_batch(
        image_paths=image_paths,
        vehicle_info={'make': 'Toyota', 'model': 'Camry', 'year': 2020}
    )
    
    for result in batch_results:
        if 'error' in result:
            print(f"{result['image_path']}: ERROR - {result['error']}")
        else:
            print(f"{result['image_path']}: {result['num_damages']} damages, ${result['total_estimated_cost']:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Vehicle Damage Assessment Pipeline")
    print("Fixed for YOLOv8 Detection Model (Bounding Boxes)")
    print("=" * 60)
    print()
    print("Pipeline ready!")
    print()
    print("Quick Start:")
    print("1. Make sure you have 'best.pt' (your trained model)")
    print("2. Create 'part_costs.csv' with part pricing")
    print("3. Run: python pipeline.py")
    print()
    print("Uncomment example_usage() below to test")
    print("=" * 60)
    
    # Uncomment to run example
    example_usage()