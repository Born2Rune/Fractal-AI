"""
LLM Adapter Layer

Adapters for commercial LLMs (GPT-4, Claude, Gemini) with cost tracking
and memory layer integration.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from .memory_layer import MemoryLayer


class CostTracker:
    """Track API costs for different models."""
    
    PRICING = {
        'gpt-4-turbo-preview': {'input': 10.0, 'output': 30.0},
        'gpt-4': {'input': 30.0, 'output': 60.0},
        'gpt-4o': {'input': 5.0, 'output': 15.0},
        'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
        'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
        'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
        'gemini-1.5-pro': {'input': 7.0, 'output': 21.0},
        'gemini-1.5-flash': {'input': 0.35, 'output': 1.05},
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pricing = self.PRICING.get(model_name, {'input': 10.0, 'output': 30.0})
        
        # Tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Dict:
        """Calculate cost for token usage."""
        input_cost = (input_tokens / 1_000_000) * self.pricing['input']
        output_cost = (output_tokens / 1_000_000) * self.pricing['output']
        total = input_cost + output_cost
        
        # Update tracking
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total,
            'model': self.model_name
        }
    
    def get_total_cost(self) -> Dict:
        """Get cumulative cost statistics."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': self.total_cost,
            'model': self.model_name
        }


class LLMAdapter:
    """
    Base adapter for commercial LLMs.
    Handles context injection and cost tracking.
    """
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.cost_tracker = CostTracker(model_name)
    
    def query_with_memory(
        self,
        memory_layer: MemoryLayer,
        query: str,
        max_context_tokens: int = 4000,
        **llm_kwargs
    ) -> Dict:
        """
        Query LLM with compressed context from memory layer.
        
        Args:
            memory_layer: MemoryLayer instance
            query: User query
            max_context_tokens: Max tokens for context
            **llm_kwargs: Additional arguments for LLM
        
        Returns:
            LLM response with cost information
        """
        # Get compressed context from memory layer
        compressed = memory_layer.query(
            query,
            max_context_tokens=max_context_tokens
        )
        
        # Build prompt with context
        system_prompt = self._build_system_prompt(compressed['context'])
        
        # Call LLM
        response = self._call_llm(
            system_prompt=system_prompt,
            user_query=query,
            **llm_kwargs
        )
        
        # Count tokens (rough estimate)
        input_tokens = compressed['tokens'] + len(query.split()) * 1.3
        output_tokens = len(response.split()) * 1.3
        
        # Track costs
        cost_info = self.cost_tracker.calculate_cost(
            int(input_tokens),
            int(output_tokens)
        )
        
        # Calculate savings (vs sending full context)
        original_cost = CostTracker(self.model_name).calculate_cost(
            int(memory_layer.stats['total_tokens_stored']),
            int(output_tokens)
        )
        
        savings = {
            'cost_this_query': cost_info['total_cost'],
            'cost_without_memory_layer': original_cost['total_cost'],
            'cost_saved': original_cost['total_cost'] - cost_info['total_cost'],
            'savings_percentage': (
                (original_cost['total_cost'] - cost_info['total_cost']) / 
                original_cost['total_cost'] * 100
                if original_cost['total_cost'] > 0 else 0
            )
        }
        
        return {
            'response': response,
            'compressed_context': compressed,
            'cost_info': cost_info,
            'savings': savings,
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with compressed context."""
        return f"""You are a helpful assistant. Use the following context to answer questions accurately.

Context:
{context}

Answer the user's question based on this context. If the answer is not in the context, say so."""
    
    def _call_llm(self, system_prompt: str, user_query: str, **kwargs) -> str:
        """Call LLM API (to be implemented by subclasses)."""
        raise NotImplementedError


class GPT4Adapter(LLMAdapter):
    """Adapter for GPT-4 API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        super().__init__(api_key, model)
        self.client = None
        
        # Lazy import and initialization
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("Warning: openai package not installed. Install with: pip install openai")
    
    def _call_llm(self, system_prompt: str, user_query: str, **kwargs) -> str:
        """Call GPT-4 API."""
        if self.client is None:
            return "Error: OpenAI client not initialized. Please install openai package."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling GPT-4: {str(e)}"


class ClaudeAdapter(LLMAdapter):
    """Adapter for Claude API."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key, model)
        self.client = None
        
        # Lazy import and initialization
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Warning: anthropic package not installed. Install with: pip install anthropic")
    
    def _call_llm(self, system_prompt: str, user_query: str, **kwargs) -> str:
        """Call Claude API."""
        if self.client is None:
            return "Error: Anthropic client not initialized. Please install anthropic package."
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_query}
                ],
                max_tokens=kwargs.get('max_tokens', 4096),
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            return response.content[0].text
        except Exception as e:
            return f"Error calling Claude: {str(e)}"


class GeminiAdapter(LLMAdapter):
    """Adapter for Gemini API."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(api_key, model)
        self.model_instance = None
        
        # Lazy import and initialization
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model_instance = genai.GenerativeModel(model)
        except ImportError:
            print("Warning: google-generativeai package not installed. Install with: pip install google-generativeai")
    
    def _call_llm(self, system_prompt: str, user_query: str, **kwargs) -> str:
        """Call Gemini API."""
        if self.model_instance is None:
            return "Error: Gemini model not initialized. Please install google-generativeai package."
        
        try:
            prompt = f"{system_prompt}\n\nUser: {user_query}"
            response = self.model_instance.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            return f"Error calling Gemini: {str(e)}"


class MemoryLayerLLM:
    """
    Unified interface combining memory layer with commercial LLMs.
    Optimized for cost and bandwidth reduction.
    """
    
    def __init__(
        self,
        session_name: str,
        llm_provider: str = 'gpt4',
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        max_context_tokens: int = 4000,
        device: str = 'cuda'
    ):
        """
        Initialize memory layer + LLM system.
        
        Args:
            session_name: Name for memory session
            llm_provider: Which LLM to use ('gpt4', 'claude', 'gemini')
            llm_api_key: API key for LLM
            llm_model: Specific model name (optional)
            max_context_tokens: Max tokens to send to LLM per query
            device: Device for memory layer
        """
        # Initialize memory layer
        self.memory_layer = MemoryLayer(
            session_name=session_name,
            enable_persistence=True,
            device=device
        )
        
        # Initialize LLM adapter
        if llm_api_key is None:
            import os
            llm_api_key = os.getenv(f'{llm_provider.upper()}_API_KEY')
            if llm_api_key is None:
                print(f"Warning: No API key provided for {llm_provider}")
        
        if llm_provider == 'gpt4':
            self.llm = GPT4Adapter(
                api_key=llm_api_key or '',
                model=llm_model or 'gpt-4-turbo-preview'
            )
        elif llm_provider == 'claude':
            self.llm = ClaudeAdapter(
                api_key=llm_api_key or '',
                model=llm_model or 'claude-3-opus-20240229'
            )
        elif llm_provider == 'gemini':
            self.llm = GeminiAdapter(
                api_key=llm_api_key or '',
                model=llm_model or 'gemini-1.5-pro'
            )
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
        
        self.max_context_tokens = max_context_tokens
        self.session_name = session_name
        self.llm_provider = llm_provider
    
    def ingest(self, context: str, metadata: Optional[Dict] = None) -> Dict:
        """Ingest large context into memory layer."""
        return self.memory_layer.ingest_context(
            context,
            metadata=metadata,
            show_progress=True
        )
    
    def chat(
        self,
        query: str,
        retrieval_strategy: str = 'hybrid',
        **llm_kwargs
    ) -> Dict:
        """
        Chat with LLM using memory layer for context.
        
        Args:
            query: User query
            retrieval_strategy: How to retrieve context
            **llm_kwargs: Additional LLM parameters
        
        Returns:
            Response with cost savings information
        """
        result = self.llm.query_with_memory(
            memory_layer=self.memory_layer,
            query=query,
            max_context_tokens=self.max_context_tokens,
            retrieval_strategy=retrieval_strategy,
            **llm_kwargs
        )
        
        return result
    
    def get_report(self) -> Dict:
        """Get comprehensive usage and savings report."""
        memory_stats = self.memory_layer.get_statistics()
        cost_stats = self.llm.cost_tracker.get_total_cost()
        
        return {
            'session_name': self.session_name,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm.model_name,
            'memory_statistics': memory_stats,
            'cost_statistics': cost_stats,
            'total_cost_saved': memory_stats['total_cost_saved'],
            'bandwidth_saved_mb': memory_stats['bandwidth_saved_mb'],
            'average_compression_ratio': f"{memory_stats['compression_ratio']:.4f}",
            'queries_processed': memory_stats['total_queries']
        }
    
    def save_session(self, metadata: Optional[Dict] = None):
        """Save memory layer session."""
        if self.memory_layer.session_manager:
            # Would save actual model state
            print(f"Session '{self.session_name}' saved")
    
    def load_session(self):
        """Load memory layer session."""
        if self.memory_layer.session_manager:
            # Would load actual model state
            print(f"Session '{self.session_name}' loaded")
