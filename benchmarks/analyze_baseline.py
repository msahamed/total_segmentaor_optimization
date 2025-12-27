import json
import numpy as np
from pathlib import Path

def analyze():
    data_path = Path("benchmarks/vanilla_ground_truth.json")
    if not data_path.exists():
        print("Results file not found.")
        return

    with open(data_path, "r") as f:
        results = json.load(f)

    # Basic stats
    latencies = [r["latency_sec"] for r in results]
    organs = [r["num_organs"] for r in results]
    
    # Calculate throughput (voxels/sec)
    throughputs = []
    for r in results:
        vols = np.prod(r["shape"])
        throughputs.append(vols / r["latency_sec"])

    # Categorical breakdown
    categories = sorted(list(set(r["category"] for r in results)))
    cat_stats = {}
    for cat in categories:
        cat_data = [r for r in results if r["category"] == cat]
        cat_latencies = [r["latency_sec"] for r in cat_data]
        cat_organs = [r["num_organs"] for r in cat_data]
        cat_vols = [np.prod(r["shape"]) for r in cat_data]
        cat_tp = [v/l for v, l in zip(cat_vols, cat_latencies)]
        
        cat_stats[cat] = {
            "count": len(cat_data),
            "avg_latency": np.mean(cat_latencies),
            "avg_organs": np.mean(cat_organs),
            "avg_throughput": np.mean(cat_tp)
        }

    # Summary report
    print("# Vanilla Baseline Analysis Report (Fast Model)")
    print(f"\n## Overall Statistics (N={len(results)})")
    print(f"- **Avg Latency**: {np.mean(latencies):.2f}s (±{np.std(latencies):.2f}s)")
    print(f"- **Min/Max Latency**: {np.min(latencies):.2f}s / {np.max(latencies):.2f}s")
    print(f"- **Avg Organs**: {np.mean(organs):.2f}")
    print(f"- **Avg Throughput**: {np.mean(throughputs)/1e6:.2f} MVox/s")

    print("\n## Categorical Breakdown")
    print("| Category | Count | Avg Latency | Avg Organs | Avg Throughput |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for cat, stats in cat_stats.items():
        print(f"| {cat} | {stats['count']} | {stats['avg_latency']:.2f}s | {stats['avg_organs']:.2f} | {stats['avg_throughput']/1e6:.2f} MVox/s |")

    # List top 5 slowest subjects
    print("\n## Top 5 Slowest Subjects")
    sorted_results = sorted(results, key=lambda x: x["latency_sec"], reverse=True)
    print("| Subject | Category | Shape | Latency |")
    print("| :--- | :--- | :--- | :--- |")
    for r in sorted_results[:5]:
        print(f"| {r['subject']} | {r['category']} | {r['shape']} | {r['latency_sec']:.2f}s |")

if __name__ == "__main__":
    analyze()
