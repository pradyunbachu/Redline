"""
Vehicle Valuation Module

Calculates cost multipliers based on vehicle make, model, year, and mileage.
Luxury vehicles, newer years, and lower mileage result in higher repair costs.
"""

from typing import Dict, Optional
import math


class VehicleValuation:
    """
    Calculates cost adjustment multipliers based on vehicle characteristics.
    """
    
    # Luxury tier classification (1 = economy, 5 = ultra-luxury)
    LUXURY_TIERS = {
        # Ultra-Luxury (Tier 5) - 1.5x to 2.0x multiplier
        'bentley': 5, 'rolls-royce': 5, 'mclaren': 5, 'ferrari': 5, 
        'lamborghini': 5, 'bugatti': 5, 'aston martin': 5, 'porsche': 5,
        'maserati': 5, 'lotus': 5,
        
        # High Luxury (Tier 4) - 1.3x to 1.5x multiplier
        'mercedes-benz': 4, 'mercedes': 4, 'bmw': 4, 'audi': 4, 
        'lexus': 4, 'acura': 4, 'infiniti': 4, 'cadillac': 4,
        'jaguar': 4, 'land rover': 4, 'range rover': 4, 'tesla': 4,
        'genesis': 4, 'lincoln': 4,
        
        # Mid-Luxury (Tier 3) - 1.1x to 1.3x multiplier
        'volvo': 3, 'volkswagen': 3, 'mazda': 3, 'subaru': 3,
        'honda': 3, 'toyota': 3, 'nissan': 3, 'hyundai': 3,
        'kia': 3, 'ford': 3, 'chevrolet': 3, 'gmc': 3,
        'dodge': 3, 'jeep': 3, 'ram': 3, 'chrysler': 3,
        'buick': 3, 'mini': 3,
        
        # Economy (Tier 2) - 0.9x to 1.0x multiplier
        'mitsubishi': 2, 'suzuki': 2, 'fiat': 2, 'smart': 2,
        
        # Budget (Tier 1) - 0.7x to 0.9x multiplier
        'dacia': 1, 'lada': 1, 'tata': 1,
    }
    
    # Premium models within brands (boosts tier by 1)
    PREMIUM_MODELS = {
        # Mercedes
        's-class': 1, 'e-class': 1, 'g-class': 1, 'amg': 1,
        'maybach': 2,  # Ultra-premium
        # BMW
        '7-series': 1, '5-series': 1, 'x7': 1, 'x5': 1, 'm-series': 1,
        # Audi
        'a8': 1, 'a6': 1, 'q7': 1, 'q8': 1, 's-line': 1, 'rs': 1,
        # Lexus
        'ls': 1, 'lx': 1, 'gs': 1, 'rc': 1,
        # Tesla
        'model s': 1, 'model x': 1, 'model 3': 1, 'model y': 1,
        # Porsche
        '911': 1, 'cayenne': 1, 'panamera': 1, 'macan': 1,
        # Toyota
        'land cruiser': 1, 'sequoia': 1, 'tundra': 1,
        # Ford
        'f-150': 0.5, 'raptor': 1, 'mustang': 0.5, 'bronco': 0.5,
        # Chevrolet
        'corvette': 1, 'tahoe': 0.5, 'suburban': 0.5,
        # General premium indicators
        'platinum': 0.5, 'limited': 0.5, 'premium': 0.5, 'luxury': 0.5,
        'sport': 0.5, 'performance': 0.5,
    }
    
    def __init__(self):
        """Initialize vehicle valuation system."""
        pass
    
    def calculate_multiplier(
        self,
        vehicle_info: Optional[Dict[str, any]] = None
    ) -> float:
        """
        Calculate cost multiplier based on vehicle characteristics.
        
        Args:
            vehicle_info: Dictionary with 'make', 'model', 'year', 'mileage'
            
        Returns:
            Multiplier (0.7 to 2.0) to apply to base repair costs
        """
        if not vehicle_info:
            return 1.0  # Default: no adjustment
        
        make = str(vehicle_info.get('make', '')).lower().strip()
        model = str(vehicle_info.get('model', '')).lower().strip()
        year = vehicle_info.get('year')
        mileage = vehicle_info.get('mileage')
        
        # Base multiplier from luxury tier
        base_multiplier = self._get_luxury_multiplier(make, model)
        
        # Adjust for year (newer = higher multiplier)
        year_multiplier = self._get_year_multiplier(year)
        
        # Adjust for mileage (lower = higher multiplier)
        mileage_multiplier = self._get_mileage_multiplier(mileage)
        
        # Combine multipliers (multiplicative)
        final_multiplier = base_multiplier * year_multiplier * mileage_multiplier
        
        # Clamp to reasonable range (0.5x to 2.5x)
        final_multiplier = max(0.5, min(2.5, final_multiplier))
        
        return round(final_multiplier, 3)
    
    def _get_luxury_multiplier(self, make: str, model: str) -> float:
        """
        Get base multiplier from luxury tier.
        
        Returns:
            Multiplier between 0.7 and 2.0
        """
        # Check make
        tier = self.LUXURY_TIERS.get(make, 3)  # Default to mid-tier
        
        # Check if model boosts tier
        model_boost = 0
        for premium_model, boost in self.PREMIUM_MODELS.items():
            if premium_model in model:
                model_boost = max(model_boost, boost)
        
        # Adjust tier based on model
        tier = min(5, tier + model_boost)
        
        # Convert tier to multiplier
        # Tier 1: 0.7x, Tier 2: 0.9x, Tier 3: 1.0x, Tier 4: 1.3x, Tier 5: 1.8x
        tier_multipliers = {
            1: 0.7,
            2: 0.9,
            3: 1.0,
            4: 1.3,
            5: 1.8
        }
        
        return tier_multipliers.get(tier, 1.0)
    
    def _get_year_multiplier(self, year: Optional[int]) -> float:
        """
        Calculate multiplier based on vehicle year.
        Newer vehicles = higher multiplier.
        
        Args:
            year: Vehicle year (e.g., 2020)
            
        Returns:
            Multiplier between 0.8 and 1.3
        """
        if year is None:
            return 1.0
        
        current_year = 2024  # Update as needed
        age = current_year - year
        
        # Newer cars (0-2 years old): 1.2x to 1.3x
        if age <= 2:
            return 1.2 + (2 - age) * 0.05  # 1.2 to 1.3
        # Recent (3-5 years): 1.1x to 1.2x
        elif age <= 5:
            return 1.1 + (5 - age) * 0.05  # 1.1 to 1.2
        # Mid-age (6-10 years): 1.0x to 1.1x
        elif age <= 10:
            return 1.0 + (10 - age) * 0.02  # 1.0 to 1.1
        # Older (11-15 years): 0.9x to 1.0x
        elif age <= 15:
            return 0.9 + (15 - age) * 0.02  # 0.9 to 1.0
        # Very old (16+ years): 0.8x to 0.9x
        else:
            return max(0.8, 1.0 - (age - 15) * 0.01)
    
    def _get_mileage_multiplier(self, mileage: Optional[int]) -> float:
        """
        Calculate multiplier based on vehicle mileage.
        Lower mileage = higher multiplier (better condition).
        
        Args:
            mileage: Vehicle mileage in miles
            
        Returns:
            Multiplier between 0.85 and 1.25
        """
        if mileage is None:
            return 1.0
        
        # Very low mileage (< 10k): 1.2x to 1.25x
        if mileage < 10000:
            return 1.2 + (10000 - mileage) / 100000 * 0.05  # Up to 1.25
        # Low mileage (10k-30k): 1.1x to 1.2x
        elif mileage < 30000:
            return 1.1 + (30000 - mileage) / 20000 * 0.1  # 1.1 to 1.2
        # Normal (30k-60k): 1.0x to 1.1x
        elif mileage < 60000:
            return 1.0 + (60000 - mileage) / 30000 * 0.1  # 1.0 to 1.1
        # Moderate (60k-100k): 0.95x to 1.0x
        elif mileage < 100000:
            return 0.95 + (100000 - mileage) / 40000 * 0.05  # 0.95 to 1.0
        # High (100k-150k): 0.9x to 0.95x
        elif mileage < 150000:
            return 0.9 + (150000 - mileage) / 50000 * 0.05  # 0.9 to 0.95
        # Very high (150k+): 0.85x to 0.9x
        else:
            return max(0.85, 0.9 - (mileage - 150000) / 100000 * 0.05)
    
    def get_valuation_info(
        self,
        vehicle_info: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Get detailed valuation information.
        
        Returns:
            Dictionary with multiplier and breakdown
        """
        if not vehicle_info:
            return {
                'multiplier': 1.0,
                'luxury_tier': 'Unknown',
                'year_adjustment': 1.0,
                'mileage_adjustment': 1.0
            }
        
        make = str(vehicle_info.get('make', '')).lower().strip()
        model = str(vehicle_info.get('model', '')).lower().strip()
        year = vehicle_info.get('year')
        mileage = vehicle_info.get('mileage')
        
        luxury_tier = self._get_luxury_tier_name(make, model)
        year_mult = self._get_year_multiplier(year)
        mileage_mult = self._get_mileage_multiplier(mileage)
        total_mult = self.calculate_multiplier(vehicle_info)
        
        return {
            'multiplier': total_mult,
            'luxury_tier': luxury_tier,
            'year_adjustment': year_mult,
            'mileage_adjustment': mileage_mult,
            'make': make,
            'model': model,
            'year': year,
            'mileage': mileage
        }
    
    def _get_luxury_tier_name(self, make: str, model: str) -> str:
        """Get human-readable luxury tier name."""
        tier = self.LUXURY_TIERS.get(make, 3)
        
        # Check model boost
        model_boost = 0
        for premium_model, boost in self.PREMIUM_MODELS.items():
            if premium_model in model:
                model_boost = max(model_boost, boost)
        
        tier = min(5, tier + model_boost)
        
        tier_names = {
            1: 'Budget',
            2: 'Economy',
            3: 'Mid-Range',
            4: 'Luxury',
            5: 'Ultra-Luxury'
        }
        
        return tier_names.get(tier, 'Mid-Range')

