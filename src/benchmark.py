import time
import torch
import numpy as np
from inference import EmotionClassifier

def benchmark(model_path="./emotion_bert_model", num_runs=1000, warmup_runs=100):
    print(f"Initializing model from {model_path}...")
    try:
        classifier = EmotionClassifier(model_path=model_path)
    except OSError:
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return

    text = "I am feeling absolutely wonderful today!"
    print(f"Benchmarking on device: {classifier.device}")
    print(f"Input text: '{text}'")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Measured runs: {num_runs}")

    # Warmup
    for _ in range(warmup_runs):
        classifier.predict(text)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        classifier.predict(text)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to ms

    latencies = np.array(latencies)
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    std = np.std(latencies)
    throughput = 1000 / avg

    print("\n--- Benchmark Results ---")
    print(f"Average Latency: {avg:.2f} ms")
    print(f"P50 Latency:     {p50:.2f} ms")
    print(f"P95 Latency:     {p95:.2f} ms")
    print(f"P99 Latency:     {p99:.2f} ms")
    print(f"Std Dev:         {std:.2f} ms")
    print(f"Throughput:      {throughput:.2f} requests/second")

if __name__ == "__main__":
    benchmark()
