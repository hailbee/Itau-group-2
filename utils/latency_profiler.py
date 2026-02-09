"""
Latency measurement framework for benchmarking model inference.

Measures runtime, memory usage, and GPU utilization for models.
"""

import torch
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class LatencyMetrics:
    """Container for latency and performance metrics."""
    model_name: str
    model_type: str
    batch_size: int
    
    # Runtime metrics (in milliseconds)
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    
    # Memory metrics (in MB)
    memory_allocated_mb: float
    memory_reserved_mb: float
    peak_memory_mb: float
    
    # CPU metrics
    cpu_memory_mb: float
    cpu_percent: float
    
    # GPU metrics (if available)
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    
    # Throughput
    throughput_samples_per_sec: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self):
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self):
        """Pretty print metrics."""
        lines = [
            f"Model: {self.model_name} ({self.model_type})",
            f"Batch Size: {self.batch_size}",
            "",
            "Runtime Metrics (ms):",
            f"  Mean: {self.mean_latency_ms:.2f}",
            f"  Min: {self.min_latency_ms:.2f}",
            f"  Max: {self.max_latency_ms:.2f}",
            f"  Std: {self.std_latency_ms:.2f}",
            "",
            "Memory Metrics (MB):",
            f"  Allocated: {self.memory_allocated_mb:.2f}",
            f"  Reserved: {self.memory_reserved_mb:.2f}",
            f"  Peak: {self.peak_memory_mb:.2f}",
            f"  CPU: {self.cpu_memory_mb:.2f}",
            "",
            "Utilization:",
            f"  CPU: {self.cpu_percent:.1f}%",
        ]
        
        if self.gpu_memory_mb is not None and self.gpu_utilization_percent is not None:
            lines.extend([
                f"  GPU Memory: {self.gpu_memory_mb:.2f}",
                f"  GPU Utilization: {self.gpu_utilization_percent:.1f}%",
            ])
        
        lines.extend([
            "",
            f"Throughput: {self.throughput_samples_per_sec:.2f} samples/sec",
        ])
        
        return "\n".join(lines)


class LatencyProfiler:
    """Profile latency and performance metrics for inference."""
    
    def __init__(self, num_runs: int = 10, warmup_runs: int = 3):
        """
        Initialize latency profiler.
        
        Args:
            num_runs: Number of inference runs to measure (excluding warmup)
            warmup_runs: Number of warmup runs before measuring
        """
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.process = psutil.Process(os.getpid())
    
    def profile_text_encoder(self, model_wrapper, texts: List[str], batch_size: int = 32) -> LatencyMetrics:
        """
        Profile text encoder model.
        
        Args:
            model_wrapper: Text encoder model wrapper with encode_text() method
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            LatencyMetrics object with detailed performance metrics
        """
        model_name = getattr(model_wrapper, 'model_name', 'unknown')
        model_type = getattr(model_wrapper.__class__, '__name__', 'Unknown')
        
        # Get initial memory state
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = model_wrapper.encode_text(texts[:batch_size])
        
        # Measure latencies
        latencies = []
        
        for _ in range(self.num_runs):
            # Record memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model_wrapper.encode_text(texts[:batch_size])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency_ms)
        
        latencies = latencies[self.warmup_runs:]  # Remove warmup runs from results
        
        # Calculate statistics
        import numpy as np
        latencies_array = np.array(latencies)
        mean_latency = float(np.mean(latencies_array))
        min_latency = float(np.min(latencies_array))
        max_latency = float(np.max(latencies_array))
        std_latency = float(np.std(latencies_array))
        throughput = (batch_size * 1000.0) / mean_latency  # samples per second
        
        # Memory metrics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            gpu_memory = None
            gpu_reserved = None
            gpu_peak = None
        
        # CPU metrics
        cpu_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        return LatencyMetrics(
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            mean_latency_ms=mean_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            std_latency_ms=std_latency,
            memory_allocated_mb=gpu_memory if gpu_memory else 0,
            memory_reserved_mb=gpu_reserved if gpu_reserved else 0,
            peak_memory_mb=gpu_peak if gpu_peak else 0,
            cpu_memory_mb=cpu_memory,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory,
            gpu_utilization_percent=None,  # Requires nvidia-ml-py
            throughput_samples_per_sec=throughput
        )

    def profile_ocr(self, ocr_tester, test_texts: List[str], include_glyph_time: bool = True) -> LatencyMetrics:
        """
        Profile OCR text extraction latency.
        
        Args:
            ocr_tester: OCRTester instance
            test_texts: List of texts to test OCR extraction on
            include_glyph_time: Whether to include text-to-glyph conversion time
            
        Returns:
            LatencyMetrics object with detailed performance metrics
        """
        from model_utils.utils.text_to_glyph import text_to_glyphs_batch
        
        model_name = 'pytesseract'
        model_type = 'OCR'
        
        # Get initial memory state
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Pre-generate glyphs if not including glyph time
        if not include_glyph_time:
            print(f"Pre-generating glyphs for {len(test_texts)} texts...")
            glyphs = text_to_glyphs_batch(test_texts, image_size=ocr_tester.glyph_size)
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            if include_glyph_time:
                glyphs = text_to_glyphs_batch(test_texts[:min(len(test_texts), 10)], image_size=ocr_tester.glyph_size)
            _ = ocr_tester._batch_extract_text(glyphs if not include_glyph_time else text_to_glyphs_batch(test_texts[:min(len(test_texts), 10)], image_size=ocr_tester.glyph_size))
        
        # Measure latencies
        latencies = []
        
        for _ in range(self.num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            if include_glyph_time:
                glyphs_batch = text_to_glyphs_batch(test_texts, image_size=ocr_tester.glyph_size)
                extracted_texts = ocr_tester._batch_extract_text(glyphs_batch)
            else:
                extracted_texts = ocr_tester._batch_extract_text(glyphs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        latencies = latencies[self.warmup_runs:]
        
        # Calculate statistics
        import numpy as np
        latencies_array = np.array(latencies)
        mean_latency = float(np.mean(latencies_array))
        min_latency = float(np.min(latencies_array))
        max_latency = float(np.max(latencies_array))
        std_latency = float(np.std(latencies_array))
        throughput = (len(test_texts) * 1000.0) / mean_latency
        
        # Memory metrics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_memory = None
            gpu_reserved = None
            gpu_peak = None
        
        # CPU metrics
        cpu_memory = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        preprocessing_note = " (including glyph generation)" if include_glyph_time else " (inference only)"
        
        return LatencyMetrics(
            model_name=model_name + preprocessing_note,
            model_type=model_type,
            batch_size=len(test_texts),
            mean_latency_ms=mean_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            std_latency_ms=std_latency,
            memory_allocated_mb=gpu_memory if gpu_memory else 0,
            memory_reserved_mb=gpu_reserved if gpu_reserved else 0,
            peak_memory_mb=gpu_peak if gpu_peak else 0,
            cpu_memory_mb=cpu_memory,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory,
            gpu_utilization_percent=None,
            throughput_samples_per_sec=throughput
        )

    def profile_image_encoder(self, model, test_texts: List[str], batch_size: int, 
                            glyph_size: Tuple[int, int], include_glyph_time: bool = True) -> LatencyMetrics:
        """
        Profile image encoder latency on glyph-based text encoding.
        
        Args:
            model: ImageEncoder model with encode_images() method
            test_texts: List of texts to encode as glyphs
            batch_size: Batch size for encoding
            glyph_size: Size of generated glyphs (H, W)
            include_glyph_time: Whether to include text-to-glyph conversion time
            
        Returns:
            LatencyMetrics object with detailed performance metrics
        """
        from model_utils.utils.text_to_glyph import text_to_glyphs_batch
        
        model_name = getattr(model, 'model_name', 'unknown')
        model_type = getattr(model.__class__, '__name__', 'ImageEncoder')
        
        # Get initial memory state
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Pre-generate glyphs if not including glyph time
        if not include_glyph_time:
            print(f"Pre-generating glyphs for {len(test_texts)} texts...")
            glyphs = text_to_glyphs_batch(test_texts, image_size=glyph_size)
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            if include_glyph_time:
                glyphs = text_to_glyphs_batch(test_texts[:min(len(test_texts), batch_size)], image_size=glyph_size)
            else:
                glyphs = glyphs[:min(len(glyphs), batch_size)]
            
            with torch.no_grad():
                _ = model.encode_images(glyphs)
        
        # Measure latencies
        latencies = []
        
        for _ in range(self.num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            if include_glyph_time:
                glyphs = text_to_glyphs_batch(test_texts, image_size=glyph_size)
                with torch.no_grad():
                    _ = model.encode_images(glyphs)
            else:
                with torch.no_grad():
                    _ = model.encode_images(glyphs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        latencies = latencies[self.warmup_runs:]
        
        # Calculate statistics
        import numpy as np
        latencies_array = np.array(latencies)
        mean_latency = float(np.mean(latencies_array))
        min_latency = float(np.min(latencies_array))
        max_latency = float(np.max(latencies_array))
        std_latency = float(np.std(latencies_array))
        throughput = (len(test_texts) * 1000.0) / mean_latency
        
        # Memory metrics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_memory = None
            gpu_reserved = None
            gpu_peak = None
        
        # CPU metrics
        cpu_memory = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        preprocessing_note = " (including glyph generation)" if include_glyph_time else " (inference only)"
        
        return LatencyMetrics(
            model_name=model_name + preprocessing_note,
            model_type=model_type,
            batch_size=len(test_texts),
            mean_latency_ms=mean_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            std_latency_ms=std_latency,
            memory_allocated_mb=gpu_memory if gpu_memory else 0,
            memory_reserved_mb=gpu_reserved if gpu_reserved else 0,
            peak_memory_mb=gpu_peak if gpu_peak else 0,
            cpu_memory_mb=cpu_memory,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory,
            gpu_utilization_percent=None,
            throughput_samples_per_sec=throughput
        )

    def profile_siamese_model(self, siamese_model, test_pairs: List[Tuple[str, str]], 
                            batch_size: int, include_glyph_time: bool = True) -> LatencyMetrics:
        """
        Profile Siamese model latency on text pair similarity.
        
        Note: The Siamese model encodes TEXT (not glyphs). The include_glyph_time parameter
        is kept for API consistency but doesn't affect measurements (always measures text encoding).
        
        Args:
            siamese_model: SiameseModelPairs instance
            test_pairs: List of (text1, text2) tuples to encode
            batch_size: Batch size for encoding
            include_glyph_time: Kept for API consistency (ignored - Siamese always uses text)
            
        Returns:
            LatencyMetrics object with detailed performance metrics
        """
        model_name = 'siamese'
        model_type = 'SiameseModel'
        
        # Get initial memory state
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Extract texts from pairs
        texts1 = [pair[0] for pair in test_pairs]
        texts2 = [pair[1] for pair in test_pairs]
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            sample_texts1 = texts1[:min(len(texts1), batch_size)]
            sample_texts2 = texts2[:min(len(texts2), batch_size)]
            
            with torch.no_grad():
                _ = siamese_model(sample_texts1, sample_texts2)
        
        # Measure latencies
        latencies = []
        
        for _ in range(self.num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = siamese_model(texts1, texts2)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        
        latencies = latencies[self.warmup_runs:]
        
        # Calculate statistics
        import numpy as np
        latencies_array = np.array(latencies)
        mean_latency = float(np.mean(latencies_array))
        min_latency = float(np.min(latencies_array))
        max_latency = float(np.max(latencies_array))
        std_latency = float(np.std(latencies_array))
        throughput = (len(test_pairs) * 1000.0) / mean_latency
        
        # Memory metrics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_memory = None
            gpu_reserved = None
            gpu_peak = None
        
        # CPU metrics
        cpu_memory = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        return LatencyMetrics(
            model_name=model_name + " (text encoding)",
            model_type=model_type,
            batch_size=batch_size,
            mean_latency_ms=mean_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            std_latency_ms=std_latency,
            memory_allocated_mb=gpu_memory if gpu_memory else 0,
            memory_reserved_mb=gpu_reserved if gpu_reserved else 0,
            peak_memory_mb=gpu_peak if gpu_peak else 0,
            cpu_memory_mb=cpu_memory,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory,
            gpu_utilization_percent=None,
            throughput_samples_per_sec=throughput
        )

    def profile_attentioncnn(
        self,
        keras_model,
        images: List,
        batch_size: int = 32,
    ) -> LatencyMetrics:
        """
        Profile latency of AttentionCNN (Keras image-based model).

        Args:
            keras_model: Loaded tf.keras model
            images: List or numpy array of preprocessed images
            batch_size: Batch size for inference

        Returns:
            LatencyMetrics
        """
        import numpy as np
        import tensorflow as tf

        model_name = "attentioncnn"
        model_type = "KerasCNN"

        # Ensure numpy array
        images = np.asarray(images)

        # Warmup
        for _ in range(self.warmup_runs):
            _ = keras_model.predict(
                images[:batch_size],
                batch_size=batch_size,
                verbose=0,
            )

        latencies = []

        for _ in range(self.num_runs):
            start = time.perf_counter()

            _ = keras_model.predict(
                images[:batch_size],
                batch_size=batch_size,
                verbose=0,
            )

            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        latencies = np.array(latencies)

        mean_latency = float(latencies.mean())
        min_latency = float(latencies.min())
        max_latency = float(latencies.max())
        std_latency = float(latencies.std())

        throughput = (batch_size * 1000.0) / mean_latency

        # CPU metrics
        cpu_memory = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)

        # TensorFlow GPU memory (safe)
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            gpu_memory = None
            if gpus:
                info = tf.config.experimental.get_memory_info("GPU:0")
                gpu_memory = info["current"] / 1024 / 1024
        except Exception:
            gpu_memory = None

        return LatencyMetrics(
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            mean_latency_ms=mean_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            std_latency_ms=std_latency,
            memory_allocated_mb=gpu_memory or 0,
            memory_reserved_mb=0,
            peak_memory_mb=0,
            cpu_memory_mb=cpu_memory,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory,
            gpu_utilization_percent=None,
            throughput_samples_per_sec=throughput,
        )
        
    def profile_mobilenetcnn(
        self,
        evaluator,             
        texts,
        batch_size: int = 32,
        ) -> LatencyMetrics:
        """
        Profile latency of MobileNet CNN (text -> glyph -> MobileNet).
    
        Args:
            evaluator: MobileNetEvaluator (has encode_texts_to_embeddings)
            texts: List[str] input texts
            batch_size: Batch size
    
        Returns:
            LatencyMetrics
        """
        import numpy as np
        import tensorflow as tf
    
        model_name = "mobilenetcnn"
        model_type = evaluator.model.__class__.__name__
    
        texts = list(map(str, texts))
        texts = texts[:batch_size]  
    
        print("Warmup runs...")
        for i in range(self.warmup_runs):
            _ = evaluator.encode_texts_to_embeddings(texts)
            print(f"  Warmup {i+1}/{self.warmup_runs} done")
    
        latencies = []
    
        for i in range(self.num_runs):
            start = time.perf_counter()
    
            _ = evaluator.encode_texts_to_embeddings(texts)
    
            end = time.perf_counter()
            latency_ms = (end - start) * 1000.0
            latencies.append(latency_ms)
    
            print(f"  Run {i+1}/{self.num_runs} finished ({latency_ms:.2f} ms)")
    
        latencies = np.array(latencies)
    
        mean_latency = float(latencies.mean())
        min_latency = float(latencies.min())
        max_latency = float(latencies.max())
        std_latency = float(latencies.std())
    
        throughput = (batch_size * 1000.0) / mean_latency
    
        cpu_memory = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)
    
        gpu_memory = None
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                info = tf.config.experimental.get_memory_info("GPU:0")
                gpu_memory = info["current"] / 1024 / 1024
        except Exception:
            gpu_memory = None
    
        return LatencyMetrics(
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            mean_latency_ms=mean_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            std_latency_ms=std_latency,
            memory_allocated_mb=gpu_memory or 0,
            memory_reserved_mb=0,
            peak_memory_mb=0,
            cpu_memory_mb=cpu_memory,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory,
            gpu_utilization_percent=None,
            throughput_samples_per_sec=throughput,
        )


def compare_models_latency(models: Dict[str, object], texts: List[str], batch_size: int = 32, num_runs: int = 10) -> List[LatencyMetrics]:
    """
    Compare latency across multiple models.
    
    Args:
        models: Dictionary mapping model names to model wrappers
        texts: List of texts to encode
        batch_size: Batch size for encoding
        num_runs: Number of runs per model
        
    Returns:
        List of LatencyMetrics objects
    """
    profiler = LatencyProfiler(num_runs=num_runs)
    results = []
    
    for model_name, model in models.items():
        print(f"Profiling {model_name}...")
        metrics = profiler.profile_text_encoder(model, texts, batch_size)
        results.append(metrics)
        print(metrics)
        print()
    
    return results



def compare_models_latency(models: Dict[str, object], texts: List[str], batch_size: int = 32, num_runs: int = 10) -> List[LatencyMetrics]:
    """
    Compare latency across multiple models.
    
    Args:
        models: Dictionary mapping model names to model wrappers
        texts: List of texts to encode
        batch_size: Batch size for encoding
        num_runs: Number of runs per model
        
    Returns:
        List of LatencyMetrics objects
    """
    profiler = LatencyProfiler(num_runs=num_runs)
    results = []
    
    for model_name, model in models.items():
        print(f"Profiling {model_name}...")
        metrics = profiler.profile_text_encoder(model, texts, batch_size)
        results.append(metrics)
        print(metrics)
        print()
    
    return results


def create_latency_report(metrics_list: List[LatencyMetrics]) -> str:
    """
    Create a formatted latency comparison report.
    
    Args:
        metrics_list: List of LatencyMetrics objects
        
    Returns:
        Formatted report string
    """
    lines = ["=" * 80]
    lines.append("LATENCY COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary table
    lines.append("Summary (sorted by mean latency):")
    lines.append("-" * 80)
    lines.append(f"{'Model':<25} {'Latency (ms)':<15} {'Throughput':<15} {'GPU Mem (MB)':<15}")
    lines.append("-" * 80)
    
    sorted_metrics = sorted(metrics_list, key=lambda m: m.mean_latency_ms)
    for metrics in sorted_metrics:
        model_name = metrics.model_name[:24]
        latency_str = f"{metrics.mean_latency_ms:.2f}±{metrics.std_latency_ms:.2f}"
        throughput_str = f"{metrics.throughput_samples_per_sec:.1f} s/s"
        gpu_mem_str = f"{metrics.gpu_memory_mb:.1f}" if metrics.gpu_memory_mb is not None else "N/A"
        
        lines.append(f"{model_name:<25} {latency_str:<15} {throughput_str:<15} {gpu_mem_str:<15}")
    
    lines.append("-" * 80)
    lines.append("")
    
    # Detailed results
    lines.append("Detailed Results:")
    lines.append("-" * 80)
    for metrics in sorted_metrics:
        lines.append(str(metrics))
        lines.append("")
    
    return "\n".join(lines)