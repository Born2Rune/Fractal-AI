"""
Interactive Chat with Fractal AI Memory Layer

Chat with Llama 3 using the optimized Fractal AI memory layer
and the 132M token corpus (WikiText-103 + 4 books).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from enhancements.memory_layer import MemoryLayer
from enhancements.llm_adapters_local import LocalLlamaAdapter

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better formatting: pip install rich")

# Configuration
LLAMA3_PATH = r"E:\Projects\Robocop\Models\llama3"
DEVICE = "cuda"
MAX_NEW_TOKENS = 200
MAX_CONTEXT_TOKENS = 3000

def print_header(text):
    if RICH_AVAILABLE:
        console.print(Panel(text, style="bold cyan"))
    else:
        print("\n" + "="*80)
        print(text)
        print("="*80 + "\n")

def print_assistant(text):
    if RICH_AVAILABLE:
        console.print(Panel(text, title="Assistant", style="green"))
    else:
        print(f"\n[Assistant]\n{text}\n")

def print_info(text):
    if RICH_AVAILABLE:
        console.print(f"[dim]{text}[/dim]")
    else:
        print(f"[Info] {text}")

print_header("Fractal AI Interactive Chat")
print("Loading corpus and models... This will take a few minutes.\n")

# Step 1: Load corpus
print("Step 1: Loading corpus...")

try:
    from datasets import load_dataset
    
    wikitext = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
    
    wikitext_parts = []
    for example in wikitext:
        text = example['text']
        if text.strip():
            wikitext_parts.append(text)
    
    wikitext_corpus = '\n'.join(wikitext_parts)
    print(f"✓ WikiText-103: {len(wikitext_corpus)/1024/1024:.1f} MB")
    
except Exception as e:
    print(f"✗ Failed to load WikiText: {e}")
    sys.exit(1)

# Load 4 books
BOOK_PATHS = [
    r"e:\Projects\Fractal AI\Amstrad Basic.txt",
    r"e:\Projects\Fractal AI\Maths.txt",
    r"e:\Projects\Fractal AI\quantum.txt",
    r"e:\Projects\Fractal AI\mechanics.txt"
]

book_parts = []
for book_path in BOOK_PATHS:
    try:
        with open(book_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            book_parts.append(content)
            book_name = Path(book_path).name
            print(f"✓ {book_name}: {len(content)/1024:.1f} KB")
    except Exception as e:
        print(f"✗ Failed to load {book_path}: {e}")

books_corpus = "\n\n=== BOOK SEPARATOR ===\n\n".join(book_parts)
combined_corpus = wikitext_corpus + "\n\n=== WIKITEXT END ===\n\n" + books_corpus

print(f"✓ Combined corpus: {len(combined_corpus)/1024/1024:.1f} MB ({len(combined_corpus)//4:,} estimated tokens)")

# Step 2: Initialize Llama 3
print("\nStep 2: Initializing Llama 3...")

try:
    llm = LocalLlamaAdapter(
        model_path=LLAMA3_PATH,
        device=DEVICE
    )
    print(f"✓ Llama 3 loaded on {DEVICE}")
except Exception as e:
    print(f"✗ Failed to load Llama 3: {e}")
    sys.exit(1)

# Step 3: Initialize Memory Layer
print("\nStep 3: Initializing Fractal AI Memory Layer...")

try:
    memory = MemoryLayer(
        session_name='interactive_chat',
        max_context_tokens=200_000_000,
        target_compression_ratio=0.00001,
        enable_persistence=False,
        device=DEVICE
    )
    print("✓ Memory layer initialized (with FAISS, caching, quantization)")
except Exception as e:
    print(f"✗ Failed to initialize memory layer: {e}")
    sys.exit(1)

# Step 4: Ingest corpus
print("\nStep 4: Ingesting corpus into memory...")
print("This will take ~3-4 minutes...")

start_time = time.time()

try:
    summary = memory.ingest_context(combined_corpus, show_progress=False)
    ingestion_time = time.time() - start_time
    
    print(f"\n✓ Ingestion complete in {ingestion_time:.1f}s")
    print(f"  Tokens: {summary['tokens_ingested']:,}")
    print(f"  Patterns: {summary['patterns_extracted']:,}")
    print(f"  FAISS index: {memory.faiss_index.ntotal:,} patterns")
    print(f"  Memory: ~{summary['patterns_extracted'] * 384 / 1024 / 1024:.0f} MB (quantized)")
except Exception as e:
    print(f"✗ Failed to ingest corpus: {e}")
    sys.exit(1)

# Step 5: Interactive chat loop
print_header("Ready to Chat!")
print("You can ask questions about:")
print("  • Wikipedia topics (history, science, people, etc.)")
print("  • Amstrad BASIC programming")
print("  • Mathematical biology")
print("  • Quantum physics (electrodynamics, mechanics)")
print("\nType 'quit' or 'exit' to end the conversation.\n")

conversation_history = []

while True:
    # Get user input
    if RICH_AVAILABLE:
        user_input = console.input("[bold blue]You:[/bold blue] ")
    else:
        user_input = input("You: ")
    
    if user_input.strip().lower() in ['quit', 'exit', 'bye']:
        print("\nGoodbye! Thanks for chatting with Fractal AI!")
        break
    
    if not user_input.strip():
        continue
    
    # Add to conversation history
    conversation_history.append(f"User: {user_input}")
    
    # Build context from conversation history
    conversation_context = "\n".join(conversation_history[-5:])  # Last 5 turns
    full_query = f"{conversation_context}\n\nUser's latest question: {user_input}\n\nProvide a helpful answer based on the knowledge base."
    
    # Query memory layer
    print_info("Searching knowledge base...")
    retrieval_start = time.time()
    
    memory_result = memory.query(full_query, max_context_tokens=MAX_CONTEXT_TOKENS)
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    print_info(f"Retrieved {memory_result['tokens']} tokens in {retrieval_time:.1f}ms (compression: {memory_result['compression_ratio']:.8f})")
    
    # Build prompt for LLM
    prompt = f"""You are a helpful assistant with access to a large knowledge base including Wikipedia, programming guides, and scientific texts.

Conversation history:
{conversation_context}

Relevant knowledge from the database:
{memory_result['context']}

User's question: {user_input}

Provide a clear, accurate, and helpful answer based on the knowledge above. If the knowledge base doesn't contain relevant information, say so."""
    
    # Generate response
    print_info("Generating response...")
    generation_start = time.time()
    
    inputs = llm.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}
    
    outputs = llm.model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=llm.tokenizer.eos_token_id
    )
    
    response = llm.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    generation_time = (time.time() - generation_start) * 1000
    
    print_info(f"Generated {len(llm.tokenizer.encode(response))} tokens in {generation_time:.0f}ms")
    
    # Display response
    print_assistant(response.strip())
    
    # Add to conversation history
    conversation_history.append(f"Assistant: {response.strip()}")
    
    print()  # Blank line for readability

# Show final statistics
print("\n" + "="*80)
print("SESSION STATISTICS")
print("="*80)
stats = memory.get_statistics()
print(f"Total queries: {stats['total_queries']}")
print(f"Average tokens per query: {stats['avg_tokens_per_query']:.0f}")
print(f"Compression ratio: {stats['compression_ratio']:.8f}")
print(f"Bandwidth saved: {stats['bandwidth_saved_mb']:.2f} MB")
print(f"Cache entries: {len(memory.query_cache)}")
