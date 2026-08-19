import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_and_select_best(csv_path, output_dir, task_name="Detection"):
    df = pd.read_csv(csv_path)
    
    # Cột metric: mAP cho Detection, Accuracy cho Classification
    metric_col = df.columns[1] 
    # Chuẩn hóa điểm: 60% Metric, 40% Speed
    df['score'] = 0.6 * (df[metric_col] / df[metric_col].max()) + 0.4 * (df['latency_ms'].min() / df['latency_ms'])
    best_row = df.loc[df['score'].idxmax()]
    
    print(f"\n🏆 Model {task_name} tốt nhất: {best_row['model']}")
    print(f"   Score: {best_row['score']:.3f} | {metric_col}: {best_row[metric_col]} | Latency: {best_row['latency_ms']}ms")
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='latency_ms', y=metric_col, s=200, hue='model')
    plt.title(f"Trade-off: {metric_col} vs Speed ({task_name})")
    plt.xlabel("Latency (ms)")
    plt.ylabel(metric_col)
    for i in range(df.shape[0]):
        plt.text(df.latency_ms[i]+1, df[metric_col][i], df.model[i], fontsize=9)
    plt.tight_layout()
    out_img = os.path.join(output_dir, f"benchmark_{task_name.lower()}_tradeoff.png")
    plt.savefig(out_img)
    print(f"📊 Biểu đồ đã lưu: {out_img}")
    
    return best_row['model']