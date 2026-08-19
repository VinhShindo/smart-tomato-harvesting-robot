import os
import yaml
from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.visualization import plot_and_select_best

CONFIG_PATH = "config/benchmark.yaml"

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    runner = BenchmarkRunner(cfg["benchmark"]["save_dir"])
    
    # 1. Benchmark Detection
    runner.run_benchmark_detection(
        cfg["detection_models"],
        cfg["benchmark"]["dataset_detection_yaml"],
        epochs=cfg["benchmark"]["epochs"]
    )
    
    # 2. Benchmark Classification
    runner.run_benchmark_classification(
        cfg["classification_models"],
        cfg["benchmark"]["dataset_classification_dir"],
        epochs=cfg["benchmark"]["epochs"]
    )
    
    # 3. Visualize & Select Best
    out_dir = cfg["benchmark"]["save_dir"]
    best_detect = plot_and_select_best(os.path.join(out_dir, "detection_results.csv"), out_dir, "Detection")
    best_classify = plot_and_select_best(os.path.join(out_dir, "classification_results.csv"), out_dir, "Classification")
    
    # 4. Save selection
    with open(os.path.join(out_dir, "best_model.txt"), "w") as f:
        f.write(f"BEST_DETECTION={best_detect}\n")
        f.write(f"BEST_CLASSIFICATION={best_classify}\n")
    print(f"✅ Benchmark hoàn tất! Xem kết quả tại {out_dir}/")

if __name__ == "__main__":
    main()