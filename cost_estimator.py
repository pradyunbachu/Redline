"""
Cost Estimation Module

Hybrid rule-based + ML approach for vehicle damage repair cost estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from vehicle_valuation import VehicleValuation


class RuleBasedCostEstimator:
    """
    Rule-based cost estimator using part cost tables and labor rates.
    """
    
    def __init__(self, cost_table_path: str, hourly_rate: float = 120.0):
        """
        Args:
            cost_table_path: Path to CSV file with part costs
            hourly_rate: Shop hourly labor rate in USD (industry standard: $100-150/hr)
        """
        self.hourly_rate = hourly_rate
        self.cost_table = pd.read_csv(cost_table_path)
        self.cost_table.set_index('part_name', inplace=True)
    
    def estimate_cost(
        self,
        part_name: str,
        damage_class: str,
        severity_class: str,
        area_ratio: float,
        replace_or_repair: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Estimate repair cost for a single damage instance.
        
        Args:
            part_name: Name of affected part (e.g., 'bumper', 'door')
            damage_class: Type of damage (e.g., 'scratch', 'dent', 'broken_glass')
            severity_class: Severity level ('minor', 'moderate', 'severe')
            area_ratio: Ratio of damage area to part area
            replace_or_repair: 'replace' or 'repair' (auto-determined if None)
            
        Returns:
            Dictionary with cost breakdown
        """
        if part_name not in self.cost_table.index:
            # Default values for unknown parts (more realistic defaults)
            part_cost = 750.0
            replace_labor_hours = 4.0
            repair_labor_hours = 2.5
        else:
            row = self.cost_table.loc[part_name]
            part_cost = float(row['part_cost'])
            replace_labor_hours = float(row['replace_labor_hours'])
            repair_labor_hours = float(row['repair_labor_hours'])
        
        # Determine replace vs repair
        if replace_or_repair is None:
            replace_or_repair = self._determine_replace_or_repair(
                damage_class, severity_class, area_ratio
            )
        
        # Base costs
        if replace_or_repair == 'replace':
            labor_hours = replace_labor_hours
            part_cost_used = part_cost
        else:  # repair
            labor_hours = repair_labor_hours
            part_cost_used = 0.0  # No part replacement
        
        # Adjust labor rate based on part complexity
        labor_rate_multiplier = self._get_labor_multiplier(part_name, damage_class)
        adjusted_hourly_rate = self.hourly_rate * labor_rate_multiplier
        labor_cost = labor_hours * adjusted_hourly_rate
        
        # Paint and materials (if needed)
        paint_cost = self._estimate_paint_cost(part_name, severity_class, area_ratio, replace_or_repair)
        
        # Shop supplies and materials (typically 5-10% of labor)
        shop_supplies = labor_cost * 0.08
        
        # Disposal fee for replaced parts
        disposal_fee = 25.0 if replace_or_repair == 'replace' else 0.0
        
        # Additional fees for complex repairs
        additional_fees = self._calculate_additional_fees(part_name, damage_class, severity_class)
        
        # Total
        total_cost = part_cost_used + labor_cost + paint_cost + shop_supplies + disposal_fee + additional_fees
        
        return {
            'part_cost': part_cost_used,
            'labor_cost': labor_cost,
            'labor_hours': labor_hours,
            'paint_cost': paint_cost,
            'shop_supplies': shop_supplies,
            'disposal_fee': disposal_fee,
            'additional_fees': additional_fees,
            'total_cost': total_cost,
            'replace_or_repair': replace_or_repair
        }
    
    def _determine_replace_or_repair(
        self,
        damage_class: str,
        severity_class: str,
        area_ratio: float
    ) -> str:
        """Determine if part should be replaced or repaired."""
        # Special cases
        if damage_class in ['broken_glass', 'shattered']:
            return 'replace'
        
        if severity_class == 'severe' and area_ratio > 0.1:
            return 'replace'
        
        if severity_class == 'minor':
            return 'repair'
        
        # Moderate damage: depends on part type and area
        if area_ratio > 0.05:
            return 'replace'
        
        return 'repair'
    
    def _estimate_paint_cost(
        self,
        part_name: str,
        severity_class: str,
        area_ratio: float,
        replace_or_repair: str
    ) -> float:
        """
        Estimate paint and materials cost.
        Paint matching, blending, and materials are expensive.
        """
        # Base paint cost per part (realistic industry prices)
        # Includes: primer, base coat, clear coat, blending, color matching
        base_paint_cost = {
            'bumper': 450.0,
            'front_bumper': 500.0,
            'rear_bumper': 450.0,
            'door': 550.0,
            'front_door': 600.0,
            'rear_door': 550.0,
            'fender': 500.0,
            'front_fender': 550.0,
            'rear_fender': 500.0,
            'hood': 650.0,
            'trunk': 550.0,
            'tailgate': 550.0,
            'quarter_panel': 600.0,
            'roof': 750.0,
            'headlight': 0.0,  # No paint for headlights
            'taillight': 0.0,
            'windshield': 0.0,
            'rear_windshield': 0.0,
            'side_window': 0.0,
            'side_mirror': 200.0,
            'grille': 300.0,
            'wheel': 250.0,
            'rim': 200.0,
            'door_handle': 150.0,
            'bumper_cover': 400.0,
            'fascia': 350.0,
        }
        
        # Get base cost (check for exact match first, then partial)
        cost = base_paint_cost.get(part_name, 0.0)
        if cost == 0.0:
            # Try to find partial match
            for key, value in base_paint_cost.items():
                if key in part_name or part_name in key:
                    cost = value
                    break
            if cost == 0.0:
                cost = 400.0  # Default for unknown parts
        
        # If repairing (not replacing), paint is usually needed
        if replace_or_repair == 'repair':
            # Repair always needs paint work
            if severity_class == 'minor':
                cost *= 0.6  # Touch-up and spot repair
            elif severity_class == 'moderate':
                cost *= 0.85  # Partial paint with blending
            # severe: full paint (100%)
        else:  # replace
            # Replacement always needs full paint
            # Add blending cost for adjacent panels (realistic)
            if part_name in ['door', 'fender', 'quarter_panel', 'hood', 'trunk']:
                cost *= 1.15  # 15% extra for blending with adjacent panels
        
        # Add color matching complexity fee (metallic, pearl, tri-coat paints)
        # This is a simplified version - real shops charge more for complex colors
        cost += 50.0  # Base color matching fee
        
        return round(cost, 2)
    
    def _get_labor_multiplier(self, part_name: str, damage_class: str) -> float:
        """
        Get labor rate multiplier based on part complexity and damage type.
        Complex parts or damage types require more skilled labor.
        """
        multiplier = 1.0
        
        # Complex parts require higher skill (and thus higher rates)
        complex_parts = ['quarter_panel', 'roof', 'hood', 'windshield', 'rear_windshield']
        if any(complex in part_name for complex in complex_parts):
            multiplier = 1.15  # 15% premium for complex work
        
        # Structural damage requires certified technicians
        if damage_class in ['structural_damage', 'frame_damage']:
            multiplier = max(multiplier, 1.25)  # 25% premium
        
        # Glass work often requires specialists
        if 'windshield' in part_name or 'window' in part_name:
            multiplier = max(multiplier, 1.1)  # 10% premium
        
        return multiplier
    
    def _calculate_additional_fees(
        self,
        part_name: str,
        damage_class: str,
        severity_class: str
    ) -> float:
        """
        Calculate additional fees for special circumstances.
        """
        fees = 0.0
        
        # Diagnostic fee for complex damage
        if severity_class == 'severe' or damage_class == 'structural_damage':
            fees += 75.0  # Diagnostic/inspection fee
        
        # Alignment check for front-end damage
        if 'front' in part_name and part_name in ['bumper', 'fender', 'headlight']:
            fees += 100.0  # Alignment check fee
        
        # Calibration fee for ADAS-equipped vehicles (common on newer cars)
        if part_name in ['windshield', 'front_bumper', 'headlight']:
            fees += 150.0  # Sensor/ADAS calibration fee
        
        # Environmental/disposal fees
        if 'glass' in part_name or 'windshield' in part_name:
            fees += 30.0  # Glass disposal fee
        
        return fees


class MLCostEstimator:
    """
    ML-based cost estimator using gradient boosted trees.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to trained model file (XGBoost/LightGBM pickle)
        """
        self.model = None
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        import pickle
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(
        self,
        features: Dict[str, float],
        part_name: str,
        vehicle_info: Optional[Dict[str, any]] = None
    ) -> float:
        """
        Predict cost using ML model.
        
        Args:
            features: Severity features dictionary
            part_name: Affected part name
            vehicle_info: Optional vehicle make/model/year/mileage
            
        Returns:
            Predicted cost in USD
        """
        if self.model is None:
            # Return placeholder if model not loaded
            return 0.0
        
        # Convert features to model input format
        # This would need to match your training data format
        feature_vector = self._features_to_vector(features, part_name, vehicle_info)
        
        prediction = self.model.predict([feature_vector])[0]
        return max(0.0, float(prediction))  # Ensure non-negative
    
    def _features_to_vector(
        self,
        features: Dict[str, float],
        part_name: str,
        vehicle_info: Optional[Dict[str, any]]
    ) -> np.ndarray:
        """Convert features dict to model input vector."""
        # This is a placeholder - actual implementation depends on model training
        # Would include one-hot encoding for part_name, vehicle make/model, etc.
        vector = [
            features.get('area_ratio', 0.0),
            features.get('compactness', 0.0),
            features.get('color_change_score', 0.0),
            features.get('depth_proxy', 0.0),
            features.get('confidence', 1.0),
        ]
        
        # Add vehicle info if available
        if vehicle_info:
            vector.extend([
                vehicle_info.get('year', 2020),
                vehicle_info.get('mileage', 50000),
            ])
        
        return np.array(vector)


class HybridCostEstimator:
    """
    Combines rule-based and ML estimators with weighted blending.
    """
    
    def __init__(
        self,
        cost_table_path: str,
        ml_model_path: Optional[str] = None,
        hourly_rate: float = 80.0,
        ml_weight: float = 0.3
    ):
        """
        Args:
            cost_table_path: Path to part cost table CSV
            ml_model_path: Optional path to trained ML model
            hourly_rate: Shop hourly labor rate
            ml_weight: Weight for ML prediction (0.0 = rule-based only, 1.0 = ML only)
        """
        # Use realistic hourly rate (industry standard is $100-150/hr)
        # Default to $120/hr if not specified
        actual_hourly_rate = max(hourly_rate, 100.0) if hourly_rate < 100 else hourly_rate
        self.rule_estimator = RuleBasedCostEstimator(cost_table_path, actual_hourly_rate)
        self.ml_estimator = MLCostEstimator(ml_model_path)
        self.ml_weight = ml_weight
        self.vehicle_valuation = VehicleValuation()
    
    def estimate(
        self,
        part_name: str,
        damage_class: str,
        severity_class: str,
        features: Dict[str, float],
        vehicle_info: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Estimate cost using hybrid approach.
        
        Args:
            part_name: Affected part name
            damage_class: Type of damage
            severity_class: Severity level
            features: Extracted severity features
            vehicle_info: Optional vehicle information
            
        Returns:
            Dictionary with cost breakdown and prediction details
        """
        # Rule-based estimate
        rule_result = self.rule_estimator.estimate_cost(
            part_name=part_name,
            damage_class=damage_class,
            severity_class=severity_class,
            area_ratio=features.get('area_ratio', 0.0)
        )
        rule_cost = rule_result['total_cost']
        
        # ML estimate
        ml_cost = self.ml_estimator.predict(features, part_name, vehicle_info)
        
        # Blend predictions
        if ml_cost > 0:  # ML model available
            blended_cost = rule_cost * (1 - self.ml_weight) + ml_cost * self.ml_weight
        else:
            blended_cost = rule_cost
            self.ml_weight = 0.0  # No ML contribution
        
        # Apply vehicle valuation multiplier
        vehicle_multiplier = self.vehicle_valuation.calculate_multiplier(vehicle_info)
        valuation_info = self.vehicle_valuation.get_valuation_info(vehicle_info)
        
        # Apply multiplier to all cost components
        adjusted_final_cost = blended_cost * vehicle_multiplier
        adjusted_rule_cost = rule_cost * vehicle_multiplier
        
        # Adjust rule breakdown costs
        adjusted_rule_breakdown = rule_result.copy()
        adjusted_rule_breakdown['part_cost'] = rule_result['part_cost'] * vehicle_multiplier
        adjusted_rule_breakdown['labor_cost'] = rule_result['labor_cost'] * vehicle_multiplier
        adjusted_rule_breakdown['paint_cost'] = rule_result['paint_cost'] * vehicle_multiplier
        adjusted_rule_breakdown['total_cost'] = rule_result['total_cost'] * vehicle_multiplier
        
        return {
            'rule_based_cost': adjusted_rule_cost,
            'ml_cost': ml_cost * vehicle_multiplier if ml_cost > 0 else 0.0,
            'final_cost': adjusted_final_cost,
            'ml_weight': self.ml_weight,
            'rule_breakdown': adjusted_rule_breakdown,
            'vehicle_multiplier': vehicle_multiplier,
            'valuation_info': valuation_info,
            'base_cost': blended_cost,  # Before multiplier
        }

