"""
Persistent Memory System for Cross-Session Storage

Implements serialization, database storage, and session management for saving
and loading complete memory states.
"""

import torch
import json
import zlib
import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle


class MemorySerializer:
    """
    Efficiently serialize/deserialize memory banks and patterns.
    Uses compression and chunking for large memory states.
    """
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
    
    def serialize_memory_banks(
        self,
        memory_banks: torch.Tensor,
        output_path: str
    ):
        """
        Serialize memory banks to disk.
        
        Format:
        - Header: metadata (num_layers, layer_size, hidden_size)
        - Body: compressed tensor data per layer
        - Footer: checksums
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # Write header
            header = {
                'num_layers': memory_banks.shape[0],
                'batch_size': memory_banks.shape[1] if memory_banks.dim() > 1 else 1,
                'layer_size': memory_banks.shape[2] if memory_banks.dim() > 2 else memory_banks.shape[1],
                'hidden_size': memory_banks.shape[3] if memory_banks.dim() > 3 else memory_banks.shape[2] if memory_banks.dim() > 2 else memory_banks.shape[1],
                'dtype': str(memory_banks.dtype),
                'version': '1.0',
                'timestamp': datetime.now().isoformat()
            }
            header_bytes = json.dumps(header).encode('utf-8')
            f.write(len(header_bytes).to_bytes(4, 'little'))
            f.write(header_bytes)
            
            # Write each layer compressed
            for layer_idx in range(memory_banks.shape[0]):
                layer_data = memory_banks[layer_idx].cpu().numpy()
                
                # Compress layer
                compressed = zlib.compress(
                    layer_data.tobytes(),
                    level=self.compression_level
                )
                
                # Write layer size and data
                f.write(len(compressed).to_bytes(4, 'little'))
                f.write(compressed)
                
                # Write checksum
                checksum = hashlib.md5(compressed).digest()
                f.write(checksum)
        
        print(f"Serialized memory banks to {output_path}")
    
    def deserialize_memory_banks(
        self,
        input_path: str,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Load memory banks from disk."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Memory banks file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            # Read header
            header_size = int.from_bytes(f.read(4), 'little')
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode('utf-8'))
            
            # Initialize tensor
            memory_banks = torch.zeros(
                header['num_layers'],
                header['batch_size'],
                header['layer_size'],
                header['hidden_size'],
                device=device
            )
            
            # Read each layer
            for layer_idx in range(header['num_layers']):
                # Read compressed data
                compressed_size = int.from_bytes(f.read(4), 'little')
                compressed = f.read(compressed_size)
                
                # Verify checksum
                checksum = f.read(16)
                if hashlib.md5(compressed).digest() != checksum:
                    raise ValueError(f"Checksum mismatch for layer {layer_idx}")
                
                # Decompress
                decompressed = zlib.decompress(compressed)
                layer_data = np.frombuffer(decompressed, dtype=np.float32)
                layer_data = layer_data.reshape(
                    header['batch_size'],
                    header['layer_size'],
                    header['hidden_size']
                )
                
                # Load to tensor
                memory_banks[layer_idx] = torch.from_numpy(layer_data).to(device)
        
        print(f"Deserialized memory banks from {input_path}")
        return memory_banks
    
    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize a single tensor to bytes."""
        tensor_np = tensor.cpu().numpy()
        tensor_bytes = tensor_np.tobytes()
        compressed = zlib.compress(tensor_bytes, level=self.compression_level)
        return compressed
    
    def deserialize_tensor(
        self,
        data: bytes,
        shape: Tuple,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Deserialize bytes to tensor."""
        decompressed = zlib.decompress(data)
        np_array = np.frombuffer(decompressed, dtype=np.float32)
        np_array = np_array.reshape(shape)
        return torch.from_numpy(np_array).to(dtype).to(device)


class PatternDatabase:
    """
    SQLite-based pattern storage for efficient querying and persistence.
    Maintains CHARM's helical indexing.
    """
    
    def __init__(self, db_path: str = 'patterns.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                rope_position INTEGER,
                helix_turn INTEGER,
                position_in_turn INTEGER,
                extracted_values TEXT,
                metadata TEXT,
                hidden_state BLOB,
                importance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices for fast lookup
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_pattern_name ON patterns(pattern_name)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_rope_position ON patterns(rope_position)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_helix_turn ON patterns(helix_turn)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_importance ON patterns(importance_score DESC)
        ''')
        
        # Complementary pairs table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS complementary_pairs (
                pattern_id_1 INTEGER,
                pattern_id_2 INTEGER,
                strand_distance INTEGER,
                FOREIGN KEY (pattern_id_1) REFERENCES patterns(id),
                FOREIGN KEY (pattern_id_2) REFERENCES patterns(id)
            )
        ''')
        
        self.conn.commit()
    
    def insert_pattern(self, pattern) -> int:
        """Insert pattern into database."""
        helix_diameter = 32
        helix_turn = pattern.rope_position // helix_diameter if hasattr(pattern, 'rope_position') else 0
        position_in_turn = pattern.rope_position % helix_diameter if hasattr(pattern, 'rope_position') else 0
        
        # Compress hidden state if present
        hidden_state_blob = None
        if hasattr(pattern, 'hidden_states') and pattern.hidden_states is not None:
            hidden_np = pattern.hidden_states.cpu().numpy()
            hidden_state_blob = zlib.compress(hidden_np.tobytes())
        
        # Serialize extracted values and metadata
        extracted_values_json = json.dumps(pattern.extracted_values) if hasattr(pattern, 'extracted_values') else '{}'
        metadata_json = json.dumps(pattern.metadata) if hasattr(pattern, 'metadata') else '{}'
        
        cursor = self.conn.execute('''
            INSERT INTO patterns (
                pattern_name, start_pos, end_pos, rope_position,
                helix_turn, position_in_turn, extracted_values,
                metadata, hidden_state, importance_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_name if hasattr(pattern, 'pattern_name') else 'unknown',
            pattern.start_pos if hasattr(pattern, 'start_pos') else 0,
            pattern.end_pos if hasattr(pattern, 'end_pos') else 0,
            pattern.rope_position if hasattr(pattern, 'rope_position') else 0,
            helix_turn,
            position_in_turn,
            extracted_values_json,
            metadata_json,
            hidden_state_blob,
            pattern.importance_score if hasattr(pattern, 'importance_score') else 0.5
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def query_by_helix_range(self, start_turn: int, end_turn: int) -> List:
        """Query patterns within a helix turn range."""
        cursor = self.conn.execute('''
            SELECT * FROM patterns
            WHERE helix_turn BETWEEN ? AND ?
            ORDER BY rope_position
        ''', (start_turn, end_turn))
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def query_by_pattern_name(self, pattern_name: str) -> List:
        """Query patterns by name."""
        cursor = self.conn.execute('''
            SELECT * FROM patterns
            WHERE pattern_name = ?
            ORDER BY rope_position
        ''', (pattern_name,))
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def query_by_importance(self, min_score: float, limit: int = 100) -> List:
        """Query most important patterns."""
        cursor = self.conn.execute('''
            SELECT * FROM patterns
            WHERE importance_score >= ?
            ORDER BY importance_score DESC
            LIMIT ?
        ''', (min_score, limit))
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def query_complementary(self, pattern_id: int) -> List:
        """Find complementary strand patterns."""
        cursor = self.conn.execute('''
            SELECT p.* FROM patterns p
            JOIN complementary_pairs cp ON p.id = cp.pattern_id_2
            WHERE cp.pattern_id_1 = ?
        ''', (pattern_id,))
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary."""
        return {
            'id': row[0],
            'pattern_name': row[1],
            'start_pos': row[2],
            'end_pos': row[3],
            'rope_position': row[4],
            'helix_turn': row[5],
            'position_in_turn': row[6],
            'extracted_values': json.loads(row[7]) if row[7] else {},
            'metadata': json.loads(row[8]) if row[8] else {},
            'hidden_state_blob': row[9],
            'importance_score': row[10],
            'created_at': row[11]
        }
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM patterns')
        total_patterns = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT COUNT(DISTINCT pattern_name) FROM patterns')
        unique_pattern_types = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT AVG(importance_score) FROM patterns')
        avg_importance = cursor.fetchone()[0] or 0.0
        
        return {
            'total_patterns': total_patterns,
            'unique_pattern_types': unique_pattern_types,
            'avg_importance_score': avg_importance
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class MemorySession:
    """
    Manages memory sessions with save/load capabilities.
    Enables resuming from previous states.
    """
    
    def __init__(self, session_dir: str = './memory_sessions'):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.serializer = MemorySerializer()
        self.pattern_db = None
    
    def save_session(
        self,
        session_name: str,
        model,
        metadata: Optional[Dict] = None
    ):
        """
        Save complete session state.
        
        Includes:
        - Memory banks
        - Pattern database
        - Model configuration
        - Processing metadata
        """
        session_path = self.session_dir / session_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving session '{session_name}'...")
        
        # Save memory banks
        if hasattr(model, 'flat_memory') and model.flat_memory.memory_banks is not None:
            memory_path = session_path / 'memory_banks.bin'
            self.serializer.serialize_memory_banks(
                model.flat_memory.memory_banks,
                str(memory_path)
            )
            print(f"  ✓ Saved memory banks")
        
        # Save patterns to database
        if hasattr(model, 'pattern_extractor'):
            pattern_db_path = session_path / 'patterns.db'
            pattern_db = PatternDatabase(str(pattern_db_path))
            
            # Get all patterns from model
            if hasattr(model.pattern_extractor, 'matches'):
                for pattern in model.pattern_extractor.matches:
                    pattern_db.insert_pattern(pattern)
            
            pattern_db.close()
            print(f"  ✓ Saved patterns database")
        
        # Save configuration
        config_path = session_path / 'config.json'
        config = {
            'session_name': session_name,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'num_layers': model.flat_memory.num_layers if hasattr(model, 'flat_memory') else 0,
                'tokens_processed': model.flat_memory.tokens_processed if hasattr(model, 'flat_memory') else 0,
                'max_patterns': model.pattern_extractor.max_patterns if hasattr(model, 'pattern_extractor') else 0,
                'compression_ratio': model.flat_memory.compression_ratio if hasattr(model, 'flat_memory') else 0.5,
            },
            'metadata': metadata or {}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Saved configuration")
        print(f"\n✓ Session '{session_name}' saved to {session_path}")
    
    def load_session(
        self,
        session_name: str,
        model,
        device: str = 'cuda'
    ) -> Dict:
        """Load session state into model."""
        session_path = self.session_dir / session_name
        
        if not session_path.exists():
            raise ValueError(f"Session '{session_name}' not found at {session_path}")
        
        print(f"\nLoading session '{session_name}'...")
        
        # Load configuration
        config_path = session_path / 'config.json'
        if not config_path.exists():
            raise ValueError(f"Session configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load memory banks
        memory_path = session_path / 'memory_banks.bin'
        if memory_path.exists() and hasattr(model, 'flat_memory'):
            memory_banks = self.serializer.deserialize_memory_banks(
                str(memory_path),
                device=device
            )
            model.flat_memory.memory_banks = memory_banks
            model.flat_memory.num_layers = config['model_config']['num_layers']
            model.flat_memory.tokens_processed = config['model_config']['tokens_processed']
            print(f"  ✓ Loaded memory banks")
        
        # Load patterns database
        pattern_db_path = session_path / 'patterns.db'
        if pattern_db_path.exists():
            self.pattern_db = PatternDatabase(str(pattern_db_path))
            print(f"  ✓ Loaded patterns database")
        
        print(f"\n✓ Session '{session_name}' loaded")
        return config['metadata']
    
    def list_sessions(self) -> List[Dict]:
        """List all available sessions."""
        sessions = []
        
        for session_path in self.session_dir.iterdir():
            if session_path.is_dir():
                config_path = session_path / 'config.json'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Get session size
                    session_size = sum(
                        f.stat().st_size
                        for f in session_path.rglob('*')
                        if f.is_file()
                    )
                    
                    sessions.append({
                        'name': session_path.name,
                        'timestamp': config.get('timestamp', 'unknown'),
                        'tokens_processed': config['model_config'].get('tokens_processed', 0),
                        'size_mb': session_size / (1024 * 1024),
                        'path': str(session_path)
                    })
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_session(self, session_name: str):
        """Delete a session."""
        import shutil
        
        session_path = self.session_dir / session_name
        if session_path.exists():
            shutil.rmtree(session_path)
            print(f"Deleted session '{session_name}'")
        else:
            print(f"Session '{session_name}' not found")
    
    def export_session(
        self,
        session_name: str,
        export_path: str,
        compress: bool = True
    ):
        """Export session to a single file."""
        import tarfile
        
        session_path = self.session_dir / session_name
        if not session_path.exists():
            raise ValueError(f"Session '{session_name}' not found")
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tar archive
        mode = 'w:gz' if compress else 'w'
        with tarfile.open(export_path, mode) as tar:
            tar.add(session_path, arcname=session_name)
        
        print(f"Exported session to {export_path}")
    
    def import_session(
        self,
        import_path: str,
        session_name: Optional[str] = None
    ):
        """Import session from exported file."""
        import tarfile
        
        import_path = Path(import_path)
        if not import_path.exists():
            raise ValueError(f"Import file not found: {import_path}")
        
        # Extract tar archive
        with tarfile.open(import_path, 'r:*') as tar:
            tar.extractall(self.session_dir)
        
        print(f"Imported session from {import_path}")
