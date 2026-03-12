"""
Utilities for latency and resource utilization benchmarking.

Provides:
- Text-to-image rendering with font management
- Resource monitoring (CPU, GPU)
- Timing utilities
- Data loading helpers
"""

import unicodedata
import psutil
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from functools import lru_cache
from typing import Tuple, List, Dict
import time


# ============================================================================
# Text Rendering - Image Generation
# ============================================================================

def _get_unicode_font(font_size: int = 14) -> ImageFont.FreeTypeFont:
    """
    Get a TrueType font from matplotlib's font cache (DejaVu Sans).
    Falls back to PIL default if not found.
    """
    try:
        path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        return ImageFont.truetype(path, font_size)
    except Exception:
        return ImageFont.load_default()


def generate_glyph_image(
    text: str,
    image_size: Tuple[int, int] = (224, 224),
    font_size: int = 14
) -> Image.Image:
    """
    Render text to a grayscale PIL Image.
    
    Args:
        text: Business name to render
        image_size: (width, height) in pixels
        font_size: Font size for rendering
        
    Returns:
        PIL Image with white text on black background
    """
    text = unicodedata.normalize('NFC', text)
    image = Image.new("RGB", image_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _get_unicode_font(font_size)
    
    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    
    # Draw white text on black background
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    
    return image


# ============================================================================
# Resource Monitoring
# ============================================================================

class ResourceMonitor:
    """Monitor CPU and GPU utilization during inference."""
    
    def __init__(self):
        self.cpu_samples = []
        self.gpu_samples = []
        self.process = psutil.Process()
        self.has_cuda = torch.cuda.is_available()
    
    def start_monitoring(self):
        """Clear buffers and prepare for monitoring."""
        self.cpu_samples = []
        self.gpu_samples = []
    
    def sample(self):
        """Take a single sample of CPU and GPU utilization."""
        # CPU: per-core usage
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        self.cpu_samples.append(cpu_per_core)
        
        # GPU: VRAM usage if available
        if self.has_cuda:
            try:
                # Use cuda memory_allocated (actual memory used)
                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                self.gpu_samples.append(gpu_mem)
                # Also trigger a sync to ensure numbers are fresh
                torch.cuda.synchronize()
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get average statistics across all samples.
        
        Returns:
            Dict with keys:
                - cpu_avg_percent: Average CPU usage across all cores
                - cpu_per_core_avg: List of average per-core usage
                - gpu_memory_avg_gb: Average GPU memory used (if CUDA available)
                - gpu_memory_max_gb: Peak GPU memory usage (if CUDA available)
        """
        stats = {}
        
        if self.cpu_samples:
            cpu_array = np.array(self.cpu_samples)
            stats['cpu_avg_percent'] = float(np.mean(cpu_array))
            stats['cpu_per_core_avg'] = [float(np.mean(cpu_array[:, i])) 
                                          for i in range(cpu_array.shape[1])]
        else:
            stats['cpu_avg_percent'] = 0.0
            stats['cpu_per_core_avg'] = []
        
        if self.gpu_samples:
            stats['gpu_memory_avg_gb'] = float(np.mean(self.gpu_samples))
            stats['gpu_memory_max_gb'] = float(np.max(self.gpu_samples))
        else:
            stats['gpu_memory_avg_gb'] = 0.0
            stats['gpu_memory_max_gb'] = 0.0
        
        return stats


class TimingContext:
    """Context manager for timing code blocks with warmup."""
    
    def __init__(self, warmup_runs: int = 1, monitor: ResourceMonitor = None):
        self.warmup_runs = warmup_runs
        self.monitor = monitor or ResourceMonitor()
        self.times = []
        self.is_warmup = True
        self.current_start = None
    
    def __call__(self, func):
        """Decorator to time a function."""
        def wrapper(*args, **kwargs):
            for _ in range(self.warmup_runs):
                self.is_warmup = True
                func(*args, **kwargs)
            
            self.is_warmup = False
            self.monitor.start_monitoring()
            self.times = []
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            self.times.append(elapsed)
            
            return result
        return wrapper
    
    def __enter__(self):
        self.times = []
        self.monitor.start_monitoring()
        self.current_start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_start is not None:
            elapsed = time.perf_counter() - self.current_start
            self.times.append(elapsed)
    
    def get_stats(self) -> Dict:
        """Get timing and resource statistics."""
        if not self.times:
            return {
                'avg_time_ms': 0.0,
                'min_time_ms': 0.0,
                'max_time_ms': 0.0,
                'std_time_ms': 0.0,
            }
        
        times_ms = [t * 1000 for t in self.times]
        return {
            'avg_time_ms': float(np.mean(times_ms)),
            'min_time_ms': float(np.min(times_ms)),
            'max_time_ms': float(np.max(times_ms)),
            'std_time_ms': float(np.std(times_ms)),
        }


# ============================================================================
# Data Loading
# ============================================================================

def load_test_data(parquet_path: str, num_samples: int = None) -> pd.DataFrame:
    """
    Load test data from parquet file.
    
    Args:
        parquet_path: Path to test_pairs_all.parquet
        num_samples: If specified, return only first num_samples rows
        
    Returns:
        DataFrame with columns: fraudulent_name, real_name, label
    """
    df = pd.read_parquet(parquet_path)
    
    # Ensure required columns exist
    required = ['fraudulent_name', 'real_name', 'label']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if num_samples is not None:
        # df = df.iloc[:num_samples]
        # randomly sample num_samples rows instead of taking the first ones to get a more representative subset
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    return df


def create_mini_dataset(num_samples: int = 10) -> pd.DataFrame:
    """
    Create a mini synthetic dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with fake name pairs and random labels
    """
    fraudulent = [f"FakeCompany{i}" for i in range(num_samples)]
    real = [f"RealCorp{i}" for i in range(num_samples)]
    labels = np.random.randint(0, 2, num_samples).tolist()
    
    return pd.DataFrame({
        'fraudulent_name': fraudulent,
        'real_name': real,
        'label': labels
    })


# ============================================================================
# Batch Processing
# ============================================================================

def create_batches(data: List, batch_size: int) -> List[List]:
    """
    Create batches from data.
    
    Args:
        data: List of items
        batch_size: Batch size
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches
