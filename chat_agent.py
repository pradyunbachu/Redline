"""
Conversational agent for answering questions about damage estimates.
Uses Groq LLM for natural, context-aware responses.
Simulates a real auto shop estimator experience.
"""

import os
from typing import Dict, List, Optional, Any
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AutoShopChatAgent:
    """
    Conversational agent that answers questions about vehicle damage estimates.
    Uses Groq LLM for natural, context-aware responses.
    Designed to simulate a real auto shop estimator conversation.
    """

    def __init__(self):
        # Initialize Groq client
        api_key = os.getenv('GROQ_API_KEY', '').strip()
        
        # Debug: Check if API key is found
        if api_key:
            print(f"✓ GROQ_API_KEY found (length: {len(api_key)})")
            try:
                self.client = Groq(api_key=api_key)
                print("✓ Groq API client initialized successfully")
            except Exception as e:
                print(f"✗ Error initializing Groq client: {e}")
                self.client = None
        else:
            self.client = None
            print("⚠ Warning: GROQ_API_KEY not found in environment variables.")
            print("  Make sure you have a .env file with GROQ_API_KEY=your_key_here")
            print("  Or set it as an environment variable: export GROQ_API_KEY=your_key_here")
        
        # Use a currently available model (llama-3.1-70b-versatile was decommissioned)
        # Options: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
        self.model = "llama-3.1-8b-instant"  # Fast and reliable model
        
        # System prompt that defines the agent's persona
        self.system_prompt = """You are a professional auto shop estimator at Redline, a vehicle damage assessment service. 
You're friendly, knowledgeable, and helpful. You speak naturally and conversationally, like you're talking to a customer in person at an auto shop.

CRITICAL RULES:
1. Keep your responses CONCISE and TO THE POINT. Aim for 2-4 sentences maximum. Be direct and clear.
2. **NEVER make up or estimate costs, prices, or numbers. ONLY use the exact numbers provided in the damage assessment data.**
3. **If cost breakdown data is provided, use those EXACT numbers. Do not calculate, estimate, or invent any costs.**
4. **When stating costs, always use the exact amounts from the provided data. If the data says "$950.00", say "$950.00" - do not change it.**

Your role:
- Answer questions about vehicle damage estimates using the EXACT damage assessment data provided
- Be brief but informative - give the key facts without unnecessary elaboration
- Reference specific numbers, parts, and costs when relevant - but ONLY use the numbers from the provided data
- If asked about timelines, give a quick estimate (e.g., "3-5 days" or "about a week")
- If asked about costs, state the EXACT amount from the data and briefly mention what it includes
- Be professional but conversational
- If you don't have specific information in the provided data, say so briefly

Tone: Professional, friendly, helpful, and CONCISE - like a busy but helpful auto shop estimator."""

    def _format_damage_context(self, damage_results: Optional[Dict]) -> str:
        """Format damage assessment data into comprehensive context for the LLM."""
        if not damage_results or not damage_results.get('damage_instances'):
            return "No damage assessment data available yet. The customer hasn't uploaded an image for assessment."
        
        context = f"=== COMPREHENSIVE DAMAGE ASSESSMENT REPORT ===\n\n"
        
        # Overall summary
        context += f"OVERALL SUMMARY:\n"
        context += f"Total Estimated Repair Cost: ${damage_results.get('total_estimated_cost', 0):,.2f}\n"
        context += f"Number of Damage Areas Detected: {damage_results.get('num_damages', 0)}\n"
        
        vehicle_multiplier = damage_results.get('vehicle_multiplier', 1.0)
        if vehicle_multiplier != 1.0:
            context += f"Vehicle Cost Adjustment Factor: {vehicle_multiplier:.2f}x (applied to all repair costs)\n"
        
        # Vehicle information
        valuation_info = damage_results.get('valuation_info', {})
        if valuation_info:
            context += f"\nVEHICLE INFORMATION:\n"
            if valuation_info.get('make'):
                context += f"Make: {valuation_info['make']}\n"
            if valuation_info.get('model'):
                context += f"Model: {valuation_info['model']}\n"
            if valuation_info.get('year'):
                context += f"Year: {valuation_info['year']}\n"
            if valuation_info.get('mileage'):
                context += f"Mileage: {valuation_info['mileage']:,} miles\n"
            if valuation_info.get('luxury_tier'):
                context += f"Luxury Classification: {valuation_info['luxury_tier']}\n"
            if valuation_info.get('year_adjustment', 1.0) != 1.0:
                context += f"Year Adjustment Factor: {valuation_info.get('year_adjustment', 1.0):.2f}x\n"
            if valuation_info.get('mileage_adjustment', 1.0) != 1.0:
                context += f"Mileage Adjustment Factor: {valuation_info.get('mileage_adjustment', 1.0):.2f}x\n"
        
        # Detailed damage instances
        damage_instances = damage_results.get('damage_instances', [])
        if damage_instances:
            context += f"\n=== DETAILED DAMAGE ANALYSIS ===\n"
            for i, instance in enumerate(damage_instances, 1):
                part_name = instance.get('part_name', 'unknown').replace('_', ' ')
                damage_class = instance.get('damage_class', 'unknown')
                severity = instance.get('severity_class', 'unknown')
                severity_score = instance.get('severity_score', 0)
                cost = instance.get('cost_estimate', {}).get('final_cost', 0)
                action = instance.get('cost_estimate', {}).get('rule_breakdown', {}).get('replace_or_repair', 'repair')
                confidence = instance.get('confidence', 0)
                
                # Bounding box information
                bbox = instance.get('bbox', {})
                bbox_coords = instance.get('bbox', [])
                
                context += f"\nDAMAGE AREA #{i}: {part_name.title()}\n"
                context += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                context += f"Damage Type: {damage_class}\n"
                context += f"Affected Part: {part_name.title()}\n"
                context += f"Severity Classification: {severity.upper()}\n"
                context += f"Severity Score: {severity_score:.2f} (on a scale where higher = more severe)\n"
                context += f"Detection Confidence: {confidence:.1%} (AI model confidence in detecting this damage)\n"
                
                if bbox_coords and len(bbox_coords) >= 4:
                    context += f"Location on Vehicle: Detected at coordinates ({bbox_coords[0]:.0f}, {bbox_coords[1]:.0f}) to ({bbox_coords[2]:.0f}, {bbox_coords[3]:.0f})\n"
                
                context += f"\nREPAIR RECOMMENDATION:\n"
                context += f"Recommended Action: {action.upper()}\n"
                if action == 'replace':
                    context += f"  → This part needs to be completely replaced due to extensive damage\n"
                else:
                    context += f"  → This part can be repaired/restored rather than replaced\n"
                
                context += f"\nCOST BREAKDOWN FOR THIS DAMAGE (USE THESE EXACT NUMBERS - DO NOT CALCULATE OR ESTIMATE):\n"
                context += f"** TOTAL COST: ${cost:,.2f} **\n"
                context += f"\nDetailed Breakdown (use these exact amounts):\n"
                
                breakdown = instance.get('cost_estimate', {}).get('rule_breakdown', {})
                cost_estimate = instance.get('cost_estimate', {})
                
                if breakdown:
                    part_cost = breakdown.get('part_cost', 0)
                    labor_cost = breakdown.get('labor_cost', 0)
                    paint_cost = breakdown.get('paint_cost', 0)
                    shop_supplies = breakdown.get('shop_supplies', 0)
                    disposal_fee = breakdown.get('disposal_fee', 0)
                    additional_fees = breakdown.get('additional_fees', 0)
                    
                    if part_cost > 0:
                        context += f"  - Part Cost: ${part_cost:,.2f} (EXACT AMOUNT - use this number)\n"
                    if labor_cost > 0:
                        # Try to get labor hours from the breakdown or estimate
                        labor_hours = breakdown.get('labor_hours') or cost_estimate.get('labor_hours', 0)
                        hourly_rate = breakdown.get('hourly_rate') or cost_estimate.get('hourly_rate', 120)
                        if labor_hours > 0:
                            context += f"  - Labor Cost: ${labor_cost:,.2f} (EXACT AMOUNT - use this number)\n"
                            context += f"    Labor Details: {labor_hours:.1f} hours @ ${hourly_rate:.0f}/hr\n"
                        else:
                            context += f"  - Labor Cost: ${labor_cost:,.2f} (EXACT AMOUNT - use this number)\n"
                    if paint_cost > 0:
                        context += f"  - Paint & Materials: ${paint_cost:,.2f} (EXACT AMOUNT - use this number)\n"
                        if breakdown.get('paint_blend_required', False):
                            context += f"    (Color matching/blending required)\n"
                    if shop_supplies > 0:
                        context += f"  - Shop Supplies: ${shop_supplies:,.2f} (EXACT AMOUNT - use this number)\n"
                    if disposal_fee > 0:
                        context += f"  - Disposal Fee: ${disposal_fee:,.2f} (EXACT AMOUNT - use this number)\n"
                    if additional_fees > 0:
                        context += f"  - Additional Fees: ${additional_fees:,.2f} (EXACT AMOUNT - use this number)\n"
                        # Check for specific fee types in the breakdown
                        if breakdown.get('diagnostic_fee', 0) > 0:
                            context += f"    (Includes diagnostic: ${breakdown.get('diagnostic_fee', 0):,.2f})\n"
                        if breakdown.get('alignment_fee', 0) > 0:
                            context += f"    (Includes alignment: ${breakdown.get('alignment_fee', 0):,.2f})\n"
                    
                    # Add a verification line
                    calculated_total = part_cost + labor_cost + paint_cost + shop_supplies + disposal_fee + additional_fees
                    context += f"\n  VERIFICATION: Part (${part_cost:,.2f}) + Labor (${labor_cost:,.2f}) + Paint (${paint_cost:,.2f}) + Supplies (${shop_supplies:,.2f}) + Disposal (${disposal_fee:,.2f}) + Additional (${additional_fees:,.2f}) = ${calculated_total:,.2f}\n"
                    context += f"  FINAL TOTAL (after vehicle adjustments): ${cost:,.2f}\n"
                    context += f"\n  IMPORTANT: When stating costs, use the EXACT numbers above. Do NOT calculate or estimate. The total is ${cost:,.2f}.\n"
        
        context += "\n=== END OF ASSESSMENT ===\n"
        context += "\nCRITICAL INSTRUCTIONS:\n"
        context += "1. Use ONLY the exact cost numbers provided above. Do NOT calculate, estimate, or invent costs.\n"
        context += "2. If a cost breakdown shows Part Cost: $X, Labor Cost: $Y, Total: $Z, use those EXACT numbers.\n"
        context += "3. Do NOT add, subtract, or recalculate costs. Use the provided totals and breakdowns as-is.\n"
        context += "4. When stating costs, quote the exact amounts from the data above.\n"
        context += "\nUse this comprehensive information to answer the customer's questions accurately using ONLY the exact numbers provided."
        
        return context

    def _format_conversation_history(self, conversation_history: List[Dict]) -> List[Dict]:
        """Format conversation history for Groq API."""
        formatted = []
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            role = "user" if msg.get('role') == 'user' else "assistant"
            content = msg.get('content', '')
            formatted.append({"role": role, "content": content})
        return formatted

    def _detect_box_question(self, message: str, damage_results: Optional[Dict]) -> Optional[List[int]]:
        """
        Detect if user is asking about specific damage boxes and return their indices.
        Returns a list of box indices (can be multiple if asking about several damages).
        """
        if not damage_results or not damage_results.get('damage_instances'):
            return None
        
        message_lower = message.lower()
        instances = damage_results['damage_instances']
        detected_indices = []
        
        # Check for explicit box numbers (1, 2, 3, etc.)
        import re
        box_number_matches = re.finditer(r'\b(box|damage|area|instance|#)?\s*(\d+)\b', message_lower)
        for match in box_number_matches:
            box_num = int(match.group(2))
            if 1 <= box_num <= len(instances):
                idx = box_num - 1  # Convert to 0-based index
                if idx not in detected_indices:
                    detected_indices.append(idx)
        
        # Check for damage types (scratch, dent, crack, etc.)
        damage_type_keywords = {
            'scratch': ['scratch', 'scratches', 'scratched'],
            'dent': ['dent', 'dented', 'dents', 'denting'],
            'crack': ['crack', 'cracks', 'cracked', 'cracking'],
            'broken': ['broken', 'break', 'broke', 'shattered'],
            'paint': ['paint', 'paint transfer', 'paint damage'],
            'structural': ['structural', 'frame damage', 'structural damage'],
        }
        
        for idx, instance in enumerate(instances):
            damage_class = instance.get('damage_class', '').lower()
            for damage_type, keywords in damage_type_keywords.items():
                if damage_type in damage_class:
                    for keyword in keywords:
                        if keyword in message_lower:
                            if idx not in detected_indices:
                                detected_indices.append(idx)
                            break
        
        # Check for part names (more comprehensive matching)
        part_keywords = {
            'bumper': ['bumper', 'front bumper', 'rear bumper', 'back bumper', 'front end', 'rear end'],
            'door': ['door', 'front door', 'rear door', 'back door', 'side door'],
            'fender': ['fender', 'front fender', 'rear fender', 'wing'],
            'hood': ['hood', 'bonnet'],
            'trunk': ['trunk', 'boot', 'rear trunk', 'tailgate'],
            'windshield': ['windshield', 'windscreen', 'front glass', 'front window'],
            'headlight': ['headlight', 'head lamp', 'front light', 'headlamp', 'headlights'],
            'taillight': ['taillight', 'tail light', 'rear light', 'brake light', 'tail lamp'],
            'mirror': ['mirror', 'side mirror', 'wing mirror'],
            'roof': ['roof', 'top'],
            'quarter panel': ['quarter panel', 'rear quarter', 'quarter'],
            'wheel': ['wheel', 'rim', 'tire'],
            'window': ['window', 'side window', 'rear window'],
        }
        
        # First, collect all part mentions in the message
        mentioned_parts = []
        for keyword, variations in part_keywords.items():
            for variation in variations:
                if variation in message_lower:
                    if keyword not in mentioned_parts:
                        mentioned_parts.append(keyword)
                    break  # Found this keyword, move to next
        
        print(f"DEBUG: Mentioned parts in message: {mentioned_parts}")
        print(f"DEBUG: Total damage instances: {len(instances)}")
        for idx, inst in enumerate(instances):
            print(f"  Instance {idx}: part_name='{inst.get('part_name', 'N/A')}', damage_class='{inst.get('damage_class', 'N/A')}'")
        
        # Then match each mentioned part to damage instances
        # Use more precise matching: check if the keyword is in the part_name
        # or if the part_name contains the keyword as a whole word
        # IMPORTANT: Check ALL mentioned parts against EACH instance to find ALL matches
        for idx, instance in enumerate(instances):
            part_name_raw = instance.get('part_name', '')
            part_name = part_name_raw.lower().replace('_', ' ').replace('-', ' ')
            # Also check damage_class and other fields that might contain part info
            damage_class = instance.get('damage_class', '').lower()
            print(f"DEBUG: Checking instance {idx}: part_name='{part_name_raw}' -> normalized='{part_name}', damage_class='{damage_class}'")
            
            # Check if this instance's part matches ANY of the mentioned parts
            instance_matched = False
            for mentioned_part in mentioned_parts:
                # More precise matching: check if mentioned_part appears as a whole word in part_name
                # or if part_name starts/ends with the mentioned_part
                part_words = part_name.split()
                matches = (mentioned_part in part_words or 
                          part_name.startswith(mentioned_part + ' ') or 
                          part_name.endswith(' ' + mentioned_part) or
                          part_name == mentioned_part)
                
                # Also check if mentioned_part is in damage_class (e.g., "headlight-damage" or "headlight-crack")
                # This handles cases where the part inference is wrong but damage_class contains the correct part
                if not matches and mentioned_part in damage_class:
                    matches = True
                    print(f"DEBUG: Match found via damage_class: '{mentioned_part}' in '{damage_class}' (part_name was '{part_name_raw}')")
                
                # Also check if any variation of the mentioned part is in damage_class
                # This is important for cases like headlight being stored under rear_fender
                if not matches:
                    for keyword, variations in part_keywords.items():
                        if keyword == mentioned_part:
                            for variation in variations:
                                if variation in damage_class:
                                    matches = True
                                    print(f"DEBUG: Match found via damage_class variation: '{variation}' in '{damage_class}' (part_name was '{part_name_raw}')")
                                    break
                            if matches:
                                break
                
                if matches:
                    print(f"DEBUG: ✓ Match found! Instance {idx} ('{part_name_raw}') matches mentioned part '{mentioned_part}'")
                    if idx not in detected_indices:
                        detected_indices.append(idx)
                        instance_matched = True
                    # Don't break - continue checking other mentioned parts for this instance
                else:
                    print(f"DEBUG: ✗ No match: '{mentioned_part}' not in '{part_name}' or '{damage_class}'")
            
            if instance_matched:
                print(f"DEBUG: Instance {idx} added to detected_indices. Current list: {detected_indices}")
        
        # Check for position words (first, second, last, etc.)
        if 'first' in message_lower or '1st' in message_lower:
            if 0 not in detected_indices:
                detected_indices.append(0)
        if 'second' in message_lower or '2nd' in message_lower:
            if len(instances) > 1 and 1 not in detected_indices:
                detected_indices.append(1)
        if 'third' in message_lower or '3rd' in message_lower:
            if len(instances) > 2 and 2 not in detected_indices:
                detected_indices.append(2)
        if 'last' in message_lower:
            last_idx = len(instances) - 1
            if last_idx >= 0 and last_idx not in detected_indices:
                detected_indices.append(last_idx)
        
        # If no specific boxes detected but message contains damage-related words, use LLM to help
        if not detected_indices and any(word in message_lower for word in ['this', 'that', 'the', 'damage', 'broken', 'problem', 'issue']):
            # Try to match by severity or cost mentions
            if 'severe' in message_lower or 'worst' in message_lower or 'most expensive' in message_lower:
                # Find the most severe or expensive damage
                max_severity = -1
                max_cost = -1
                best_idx = None
                for idx, instance in enumerate(instances):
                    severity_score = instance.get('severity_score', 0)
                    cost = instance.get('cost_estimate', {}).get('final_cost', 0)
                    if severity_score > max_severity or cost > max_cost:
                        max_severity = severity_score
                        max_cost = cost
                        best_idx = idx
                if best_idx is not None:
                    detected_indices.append(best_idx)
            elif 'minor' in message_lower or 'small' in message_lower or 'cheapest' in message_lower:
                # Find the least severe or cheapest damage
                min_severity = float('inf')
                min_cost = float('inf')
                best_idx = None
                for idx, instance in enumerate(instances):
                    severity_score = instance.get('severity_score', 0)
                    cost = instance.get('cost_estimate', {}).get('final_cost', 0)
                    if severity_score < min_severity or cost < min_cost:
                        min_severity = severity_score
                        min_cost = cost
                        best_idx = idx
                if best_idx is not None:
                    detected_indices.append(best_idx)
        
        # If still no matches, check for "all" or "both" or "each"
        if 'all' in message_lower or 'both' in message_lower or 'each' in message_lower or 'every' in message_lower:
            return list(range(len(instances)))
        
        # If we have some matches, return them
        if detected_indices:
            return detected_indices
        
        # Last resort: Try to match by comparing message words to damage descriptions
        # This helps catch cases like "the scratch on the bumper" when we have a scratch on bumper
        message_words = set(message_lower.split())
        
        # If message contains damage-related words, try harder to match
        damage_indicators = ['damage', 'broken', 'problem', 'issue', 'wrong', 'this', 'that', 'the', 'scratch', 'dent', 'crack']
        has_damage_indicator = any(word in message_lower for word in damage_indicators)
        
        if has_damage_indicator:
            for idx, instance in enumerate(instances):
                # Build a description of this damage
                part_name = instance.get('part_name', '').lower()
                damage_class = instance.get('damage_class', '').lower()
                severity = instance.get('severity_class', '').lower()
                
                # Check if message words match this damage's characteristics
                part_words = set(part_name.split())
                damage_words = set(damage_class.split())
                severity_words = set(severity.split())
                
                # More aggressive matching: check if any word from message appears in damage description
                # or if damage description words appear in message
                part_match = any(word in part_name for word in message_words if len(word) > 2) or any(word in message_lower for word in part_words if len(word) > 2)
                damage_match = any(word in damage_class for word in message_words if len(word) > 2) or any(word in message_lower for word in damage_words if len(word) > 2)
                
                if part_match or damage_match:
                    if idx not in detected_indices:
                        detected_indices.append(idx)
        
        # If still no matches and message is clearly about a specific damage (not general questions),
        # and there's only one damage instance, assume they're asking about it
        if not detected_indices and len(instances) == 1:
            # Only do this if message seems specific (not "what's the total cost" etc.)
            general_questions = ['total', 'cost', 'price', 'how much', 'estimate', 'summary', 'overall']
            if not any(q in message_lower for q in general_questions):
                detected_indices.append(0)
        
        # Always return a list, even if empty
        print(f"DEBUG: Final detected_indices: {detected_indices}")
        return detected_indices if detected_indices else None

    def generate_response(
        self,
        message: str,
        conversation_history: List[Dict],
        damage_results: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate a response to user's message using Groq LLM.
        
        Args:
            message: User's message
            conversation_history: Previous messages in conversation
            damage_results: Current damage assessment results
            
        Returns:
            Dictionary with 'response' string and optional 'box_index'
        """
        # Detect if asking about specific boxes (can be multiple)
        box_indices = self._detect_box_question(message, damage_results)
        if box_indices:
            print(f"DEBUG: Detected box indices: {box_indices} (type: {type(box_indices)})")
            if damage_results and damage_results.get('damage_instances'):
                instances = damage_results.get('damage_instances', [])
                for idx in box_indices:
                    if 0 <= idx < len(instances):
                        instance = instances[idx]
                        print(f"  - Box {idx}: {instance.get('part_name', 'Unknown')} - {instance.get('damage_class', 'Unknown')}")
        else:
            print(f"DEBUG: No box indices detected")
        
        # If no Groq API key, use fallback
        if not self.client:
            print("⚠ DEBUG: Using fallback - Groq client is None. Check your GROQ_API_KEY.")
            response = self._fallback_response(message, damage_results)
            # Ensure box_indices is always a list
            if box_indices is None:
                box_indices = []
            elif not isinstance(box_indices, list):
                box_indices = [box_indices]
            print(f"DEBUG: Fallback returning box_indices: {box_indices}")
            return {'response': response, 'box_indices': box_indices}
        
        try:
            # Format context
            damage_context = self._format_damage_context(damage_results)
            
            # If asking about specific boxes, add more detail about those boxes
            if box_indices and damage_results:
                instances = damage_results.get('damage_instances', [])
                if len(box_indices) == 1:
                    # Single box - provide detailed context
                    box_index = box_indices[0]
                    if 0 <= box_index < len(instances):
                        instance = instances[box_index]
                        box_context = f"\n\n=== SPECIFIC DAMAGE AREA #{box_index + 1} DETAILS ===\n"
                        box_context += f"This is the {box_index + 1}{'st' if box_index == 0 else 'nd' if box_index == 1 else 'rd' if box_index == 2 else 'th'} damage area.\n"
                        box_context += f"Damage Type: {instance.get('damage_class', 'Unknown')}\n"
                        box_context += f"Severity: {instance.get('severity_class', 'Unknown')} (score: {instance.get('severity_score', 0):.2f})\n"
                        box_context += f"Confidence: {instance.get('confidence', 0):.2f}\n"
                        box_context += f"** TOTAL COST: ${instance.get('cost_estimate', {}).get('final_cost', 0):,.2f} **\n"
                        
                        # Add detailed cost breakdown for this specific box
                        cost_breakdown = instance.get('cost_estimate', {}).get('rule_breakdown', {})
                        if cost_breakdown:
                            box_context += f"\nCOST BREAKDOWN (USE THESE EXACT NUMBERS):\n"
                            if cost_breakdown.get('part_cost', 0) > 0:
                                box_context += f"  Part Cost: ${cost_breakdown.get('part_cost', 0):,.2f}\n"
                            if cost_breakdown.get('labor_cost', 0) > 0:
                                box_context += f"  Labor Cost: ${cost_breakdown.get('labor_cost', 0):,.2f}\n"
                            if cost_breakdown.get('paint_cost', 0) > 0:
                                box_context += f"  Paint & Materials: ${cost_breakdown.get('paint_cost', 0):,.2f}\n"
                            if cost_breakdown.get('shop_supplies', 0) > 0:
                                box_context += f"  Shop Supplies: ${cost_breakdown.get('shop_supplies', 0):,.2f}\n"
                            if cost_breakdown.get('disposal_fee', 0) > 0:
                                box_context += f"  Disposal Fee: ${cost_breakdown.get('disposal_fee', 0):,.2f}\n"
                            if cost_breakdown.get('additional_fees', 0) > 0:
                                box_context += f"  Additional Fees: ${cost_breakdown.get('additional_fees', 0):,.2f}\n"
                            box_context += f"  TOTAL: ${instance.get('cost_estimate', {}).get('final_cost', 0):,.2f}\n"
                        
                        damage_context += box_context
                        damage_context += "\nCRITICAL: The user is asking specifically about THIS damage area. Use ONLY the exact cost numbers provided above. Do NOT calculate, estimate, or make up any costs. The total cost is ${:,.2f}.\n".format(instance.get('cost_estimate', {}).get('final_cost', 0))
                else:
                    # Multiple boxes - provide context for all
                    box_context = f"\n\n=== SPECIFIC DAMAGE AREAS REQUESTED ===\n"
                    box_context += f"The user is asking about {len(box_indices)} specific damage area(s). Here are the details for EACH one:\n\n"
                    for i, box_index in enumerate(box_indices):
                        if 0 <= box_index < len(instances):
                            instance = instances[box_index]
                            box_context += f"DAMAGE AREA #{box_index + 1}:\n"
                            box_context += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            box_context += f"Part: {instance.get('part_name', 'Unknown')}\n"
                            box_context += f"Damage Type: {instance.get('damage_class', 'Unknown')}\n"
                            box_context += f"Severity: {instance.get('severity_class', 'Unknown')} (score: {instance.get('severity_score', 0):.2f})\n"
                            box_context += f"** TOTAL COST: ${instance.get('cost_estimate', {}).get('final_cost', 0):,.2f} **\n"
                            
                            # Add cost breakdown for each box
                            cost_breakdown = instance.get('cost_estimate', {}).get('rule_breakdown', {})
                            if cost_breakdown:
                                box_context += f"\nCOST BREAKDOWN (USE THESE EXACT NUMBERS):\n"
                                if cost_breakdown.get('part_cost', 0) > 0:
                                    box_context += f"  Part Cost: ${cost_breakdown.get('part_cost', 0):,.2f}\n"
                                if cost_breakdown.get('labor_cost', 0) > 0:
                                    box_context += f"  Labor Cost: ${cost_breakdown.get('labor_cost', 0):,.2f}\n"
                                if cost_breakdown.get('paint_cost', 0) > 0:
                                    box_context += f"  Paint & Materials: ${cost_breakdown.get('paint_cost', 0):,.2f}\n"
                                if cost_breakdown.get('shop_supplies', 0) > 0:
                                    box_context += f"  Shop Supplies: ${cost_breakdown.get('shop_supplies', 0):,.2f}\n"
                                if cost_breakdown.get('disposal_fee', 0) > 0:
                                    box_context += f"  Disposal Fee: ${cost_breakdown.get('disposal_fee', 0):,.2f}\n"
                                if cost_breakdown.get('additional_fees', 0) > 0:
                                    box_context += f"  Additional Fees: ${cost_breakdown.get('additional_fees', 0):,.2f}\n"
                                box_context += f"  TOTAL: ${instance.get('cost_estimate', {}).get('final_cost', 0):,.2f}\n"
                            
                            box_context += "\n"
                    damage_context += box_context
                    damage_context += f"\nCRITICAL: The user is asking about THESE {len(box_indices)} SPECIFIC damage areas. Provide information about EACH one separately. Use ONLY the exact cost numbers provided above for each damage area. Do NOT confuse or mix up the damage areas.\n"
            
            conversation_messages = self._format_conversation_history(conversation_history)
            
            # Build messages for Groq
            # Start with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history (all previous messages, excluding current)
            # The frontend sends history WITHOUT the current message, so we can add all
            if conversation_messages:
                messages.extend(conversation_messages)
            
            # Add the current question with full damage context
            user_content = f"{damage_context}\n\nCustomer Question: {message}\n\nREMINDER: Use ONLY the exact cost numbers from the data above. Do NOT calculate or estimate. If the data shows a total cost, use that exact number."
            messages.append({"role": "user", "content": user_content})
            
            print(f"DEBUG: Calling Groq API with {len(messages)} messages")
            
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200,  # Slightly more tokens for detailed box explanations
                top_p=0.9
            )
            
            response_text = completion.choices[0].message.content.strip()
            print(f"✓ Groq API response received: {response_text[:100]}...")
            
            # Ensure box_indices is always a list (even if empty or None)
            if box_indices is None:
                box_indices = []
            elif not isinstance(box_indices, list):
                box_indices = [box_indices]
            
            print(f"DEBUG: Returning box_indices: {box_indices} (type: {type(box_indices)}, length: {len(box_indices)})")
            return {'response': response_text, 'box_indices': box_indices}
            
        except Exception as e:
            print(f"✗ Groq API error: {e}")
            import traceback
            traceback.print_exc()
            response = self._fallback_response(message, damage_results)
            # Ensure box_indices is always a list
            if box_indices is None:
                box_indices = []
            elif not isinstance(box_indices, list):
                box_indices = [box_indices]
            return {'response': response, 'box_indices': box_indices}

    def _fallback_response(self, message: str, damage_results: Optional[Dict]) -> str:
        """Fallback response when Groq is not available."""
        message_lower = message.lower().strip()
        
        # Simple fallback for common questions
        if any(word in message_lower for word in ["how long", "time", "timeline", "when", "take"]):
            if damage_results and damage_results.get('damage_instances'):
                num = len(damage_results.get('damage_instances', []))
                if num > 3:
                    return f"Based on the {num} damage areas detected, I'd estimate 5-10 business days for completion. This includes parts ordering, repair work, paint matching, and quality checks."
                elif num > 1:
                    return f"With {num} damage areas to address, I'd estimate 3-7 business days for completion."
                else:
                    return "I'd estimate 2-5 business days for completion of this repair."
            return "I'd need to see the damage assessment to provide a timeline estimate."
        
        if any(word in message_lower for word in ["cost", "price", "how much", "expensive"]):
            if damage_results:
                cost = damage_results.get('total_estimated_cost', 0)
                return f"The total estimated repair cost is ${cost:,.2f}. This includes parts, labor, paint, and all associated fees."
            return "I'd need to see the damage assessment to provide cost information."
        
        return "I'm here to help answer questions about your damage estimate. Could you be more specific about what you'd like to know?"
