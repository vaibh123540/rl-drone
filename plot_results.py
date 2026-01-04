import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Configuration
CSV_PATH = "runs/drone_ppo_log.csv"
OUTPUT_DIR = "plots"
SMOOTHING_WINDOW = 50 

def main():
    """
    Read training log CSV and generate performance plots for return and accuracy.
    """
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(CSV_PATH)

    # Filter out initialization sentinel values for plotting
    df.loc[df['best_eval'] < -1e10, 'best_eval'] = np.nan
    
    # Plot 1: Returns
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    sns.lineplot(data=df, x="seen_steps", y="ep_return_ema", linewidth=2.5, color="#4A90E2", label="Average Return")
    sns.lineplot(data=df, x="seen_steps", y="best_eval", linewidth=1.5, color="#E24A4A", linestyle="--", label="Peak Eval Score")

    plt.title("Agent Performance Over Training (1M Steps)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/training_return.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

    # Plot 2: Hit Rate
    plt.figure(figsize=(10, 6))
    
    df['smooth_hit_rate'] = df['shot_hit_rate'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    
    sns.lineplot(data=df, x="seen_steps", y="smooth_hit_rate", linewidth=2.5, color="#50C878", label="Hit Rate (Smoothed)")
    
    plt.title("Shooting Accuracy Over Time", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Hit Rate (0.0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.0) 
    plt.legend()
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/hit_rate.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    main()