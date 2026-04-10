"""
Local LLM Adapters for Fractal AI

Adapters for running Fractal AI with local models (Llama, etc.) using transformers.
"""

import torch
from typing import Dict, Optional
from pathlib import Path


class LocalLlamaAdapter:
    """
    Adapter for local Llama models using transformers.
    Works with Llama 2, Llama 3, and compatible models.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize local Llama adapter.
        
        Args:
            model_path: Path to local model directory
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        
        print(f"Loading Llama model from {model_path}...")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✓ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def query_with_memory(
        self,
        memory_layer,
        query: str,
        max_context_tokens: int = 4000,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict:
        """
        Query local LLM with compressed context from memory layer.
        
        Args:
            memory_layer: MemoryLayer instance
            query: User query
            max_context_tokens: Max tokens for context
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        
        Returns:
            Response with timing information
        """
        import time
        
        # Get compressed context from memory layer
        start_time = time.time()
        compressed = memory_layer.query(
            query,
            max_context_tokens=max_context_tokens
        )
        retrieval_time = time.time() - start_time
        
        # Build prompt with context
        system_prompt = self._build_system_prompt(compressed['context'])
        full_prompt = f"{system_prompt}\n\nUser: {query}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_context_tokens
        ).to(self.model.device)
        
        input_tokens = inputs.input_ids.shape[1]
        
        # Generate
        generation_start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        generation_time = time.time() - generation_start
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        
        return {
            'response': response.strip(),
            'compressed_context': compressed,
            'timing': {
                'retrieval_ms': retrieval_time * 1000,
                'generation_ms': generation_time * 1000,
                'total_ms': (retrieval_time + generation_time) * 1000,
                'tokens_per_second': output_tokens / generation_time if generation_time > 0 else 0
            },
            'tokens': {
                'input': input_tokens,
                'output': output_tokens,
                'context_compressed': compressed['tokens'],
                'compression_ratio': compressed['compression_ratio']
            }
        }
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with compressed context."""
        if context:
            return f"""You are a helpful assistant. Use the following context to answer questions accurately.

Context:
{context}

Answer the user's question based on this context. If the answer is not in the context, say so."""
        else:
            return "You are a helpful assistant. Answer the user's question accurately."
    
    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        """Simple generation without memory layer."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


class LocalLLMBenchmark:
    """
    Benchmark Fractal AI with local LLM.
    Measures retrieval speed, generation quality, and cost savings.
    """
    
    def __init__(self, llm_adapter, memory_layer):
        self.llm = llm_adapter
        self.memory = memory_layer
        self.results = []
    
    def run_query(self, query: str, max_context_tokens: int = 4000, max_new_tokens: int = 256) -> Dict:
        """Run a single query and collect metrics."""
        result = self.llm.query_with_memory(
            memory_layer=self.memory,
            query=query,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens
        )
        
        self.results.append({
            'query': query,
            'response': result['response'],
            'retrieval_ms': result['timing']['retrieval_ms'],
            'generation_ms': result['timing']['generation_ms'],
            'total_ms': result['timing']['total_ms'],
            'tokens_per_second': result['timing']['tokens_per_second'],
            'compression_ratio': result['tokens']['compression_ratio']
        })
        
        return result
    
    def get_summary(self) -> Dict:
        """Get benchmark summary statistics."""
        if not self.results:
            return {}
        
        return {
            'total_queries': len(self.results),
            'avg_retrieval_ms': sum(r['retrieval_ms'] for r in self.results) / len(self.results),
            'avg_generation_ms': sum(r['generation_ms'] for r in self.results) / len(self.results),
            'avg_total_ms': sum(r['total_ms'] for r in self.results) / len(self.results),
            'avg_tokens_per_second': sum(r['tokens_per_second'] for r in self.results) / len(self.results),
            'avg_compression_ratio': sum(r['compression_ratio'] for r in self.results) / len(self.results),
        }
