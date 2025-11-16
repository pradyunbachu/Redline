"""
Pipecat-based conversational agent for vehicle damage assessment.
Integrates with the existing Flask backend for real-time conversational AI.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Pipecat, but handle compatibility issues gracefully
# Pipecat requires Python 3.10+ due to type hint syntax (str | None)
import sys

PIPECAT_AVAILABLE = False
# Check Python version FIRST before attempting any imports
if sys.version_info < (3, 10):
    print("⚠ Pipecat requires Python 3.10+, but you're using Python {}.{}".format(
        sys.version_info.major, sys.version_info.minor))
    print("  Pipecat features will be disabled. Using standard Groq agent instead.")
    PIPECAT_AVAILABLE = False
    # Create dummy classes to prevent import errors
    TextFrame = None
    LLMMessagesFrame = None
    LLMResponseAggregator = None
    GroqLLMService = None
    Pipeline = None
    PipelineRunner = None
    TransportFrame = None
else:
    try:
        from pipecat.frames.frames import TextFrame, LLMMessagesFrame
        from pipecat.processors.aggregators.llm_response import LLMResponseAggregator
        from pipecat.services.groq import GroqLLMService
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.transports.base import TransportFrame
        PIPECAT_AVAILABLE = True
        print("✓ Pipecat imported successfully")
    except (ImportError, TypeError, SyntaxError) as e:
        print(f"⚠ Pipecat not available: {e}")
        print("  Using standard Groq agent instead. Pipecat requires Python 3.10+.")
        PIPECAT_AVAILABLE = False
        # Create dummy classes to prevent import errors
        TextFrame = None
        LLMMessagesFrame = None
        LLMResponseAggregator = None
        GroqLLMService = None
        Pipeline = None
        PipelineRunner = None
        TransportFrame = None


class PipecatAutoShopAgent:
    """
    Pipecat-based conversational agent for vehicle damage assessment.
    Provides real-time conversational AI capabilities.
    """
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY', '').strip()
        
        if not self.api_key:
            print("⚠ Warning: GROQ_API_KEY not found. Pipecat agent will not work.")
            self.service = None
            self.pipeline = None
            return
        
        if not PIPECAT_AVAILABLE:
            print("⚠ Pipecat dependencies not fully available. Using fallback mode.")
            self.service = None
            self.pipeline = None
            return
        
        # Only try to use Pipecat classes if they're actually available
        if GroqLLMService is None:
            print("⚠ Pipecat classes not available. Using fallback mode.")
            self.service = None
            self.pipeline = None
            return
        
        try:
            # Initialize Groq LLM service for Pipecat
            self.service = GroqLLMService(
                api_key=self.api_key,
                model="llama-3.1-8b-instant"
            )
            
            # Create response aggregator
            self.aggregator = LLMResponseAggregator()
            
            # System prompt for auto shop estimator
            self.system_prompt = """You are a professional auto shop estimator at Redline, a vehicle damage assessment service. 
You're friendly, knowledgeable, and helpful. You speak naturally and conversationally, like you're talking to a customer in person at an auto shop.

CRITICAL: Keep your responses CONCISE and TO THE POINT. Aim for 2-4 sentences maximum. Be direct and clear.

Your role:
- Answer questions about vehicle damage estimates using the damage assessment data provided
- Be brief but informative - give the key facts without unnecessary elaboration
- Reference specific numbers, parts, and costs when relevant
- If asked about timelines, give a quick estimate (e.g., "3-5 days" or "about a week")
- If asked about costs, state the amount clearly and briefly mention what it includes
- Be professional but conversational
- If you don't have specific information, say so briefly

Tone: Professional, friendly, helpful, and CONCISE - like a busy but helpful auto shop estimator."""
            
            print("✓ Pipecat agent initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing Pipecat agent: {e}")
            self.service = None
            self.pipeline = None
    
    def _format_damage_context(self, damage_results: Optional[Dict]) -> str:
        """Format damage assessment data into context for the LLM."""
        if not damage_results or not damage_results.get('damage_instances'):
            return "No damage assessment data available yet. The customer hasn't uploaded an image for assessment."
        
        context = f"=== COMPREHENSIVE DAMAGE ASSESSMENT REPORT ===\n\n"
        context += f"Total Estimated Repair Cost: ${damage_results.get('total_estimated_cost', 0):,.2f}\n"
        context += f"Number of Damage Areas Detected: {damage_results.get('num_damages', 0)}\n"
        
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
        
        damage_instances = damage_results.get('damage_instances', [])
        if damage_instances:
            context += f"\n=== DETAILED DAMAGE ANALYSIS ===\n"
            for i, instance in enumerate(damage_instances, 1):
                part_name = instance.get('part_name', 'unknown').replace('_', ' ')
                severity = instance.get('severity_class', 'unknown')
                cost = instance.get('cost_estimate', {}).get('final_cost', 0)
                action = instance.get('cost_estimate', {}).get('rule_breakdown', {}).get('replace_or_repair', 'repair')
                
                context += f"\nDAMAGE AREA #{i}: {part_name.title()}\n"
                context += f"Severity: {severity.upper()}\n"
                context += f"Estimated Cost: ${cost:,.2f}\n"
                context += f"Action: {action.upper()}\n"
        
        return context
    
    async def process_message_async(
        self,
        message: str,
        conversation_history: List[Dict],
        damage_results: Optional[Dict] = None
    ) -> str:
        """
        Process a message asynchronously using Pipecat.
        
        Args:
            message: User's message
            conversation_history: Previous messages
            damage_results: Current damage assessment results
            
        Returns:
            Response string
        """
        if not self.service:
            return "Pipecat agent is not available. Please check your configuration."
        
        try:
            # Format context
            damage_context = self._format_damage_context(damage_results)
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history
            for msg in conversation_history[-10:]:
                role = "user" if msg.get('role') == 'user' else "assistant"
                messages.append({"role": role, "content": msg.get('content', '')})
            
            # Add current message with context
            user_content = f"{damage_context}\n\nCustomer Question: {message}"
            messages.append({"role": "user", "content": user_content})
            
            # Process with Pipecat
            # Note: This is a simplified version. Full Pipecat integration would use
            # a proper pipeline with frames, but for Flask integration, we'll use
            # the service directly
            response_text = ""
            
            # For now, use the service's chat completion directly
            # (Pipecat's full pipeline is designed for real-time voice, which we'll integrate later)
            from groq import Groq
            groq_client = Groq(api_key=self.api_key)
            
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_tokens=150  # Keep responses concise
            )
            
            response_text = response.choices[0].message.content.strip()
            return response_text
            
        except Exception as e:
            print(f"Error in Pipecat agent: {e}")
            import traceback
            traceback.print_exc()
            return f"I'm having trouble processing that. Please try again. Error: {str(e)}"
    
    def process_message(
        self,
        message: str,
        conversation_history: List[Dict],
        damage_results: Optional[Dict] = None
    ) -> str:
        """
        Synchronous wrapper for process_message_async.
        For Flask integration.
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                # For Flask, we'll use the synchronous Groq client instead
                return self._process_message_sync(message, conversation_history, damage_results)
            else:
                return loop.run_until_complete(
                    self.process_message_async(message, conversation_history, damage_results)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self.process_message_async(message, conversation_history, damage_results)
            )
    
    def _process_message_sync(
        self,
        message: str,
        conversation_history: List[Dict],
        damage_results: Optional[Dict] = None
    ) -> str:
        """Synchronous processing for Flask compatibility."""
        if not self.api_key:
            return "Groq API key not configured."
        
        try:
            from groq import Groq
            groq_client = Groq(api_key=self.api_key)
            
            # Format context
            damage_context = self._format_damage_context(damage_results)
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history
            for msg in conversation_history[-10:]:
                role = "user" if msg.get('role') == 'user' else "assistant"
                messages.append({"role": role, "content": msg.get('content', '')})
            
            # Add current message with context
            user_content = f"{damage_context}\n\nCustomer Question: {message}"
            messages.append({"role": "user", "content": user_content})
            
            # Get response
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_tokens=150  # Keep responses concise
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in Pipecat agent (sync): {e}")
            import traceback
            traceback.print_exc()
            return f"I'm having trouble processing that. Please try again."

