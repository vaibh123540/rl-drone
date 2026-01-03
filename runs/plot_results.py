import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
CSV_PATH = "runs/drone_ppo_log.csv"  # Check this path matches your file
OUTPUT_DIR = "plots"
SMOOTHING_WINDOW = 50  # How many updates to average over for smooth lines

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read data
    df = pd.read_csv(CSV_PATH)
    
    # Create the Return Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    # We plot the Exponential Moving Average (EMA) from your log as it's already smoothed,
    # but we can also smooth the raw 'mean_step_reward' * 128 (approx episode return) if needed.
    # Let's stick to the 'ep_return_ema' you logged, as it's the most reliable metric.
    sns.lineplot(data=df, x="seen_steps", y="ep_return_ema", linewidth=2.5, color="#4A90E2", label="Average Return")
    
    # Overlay Best Eval scores if available
    if "best_eval" in df.columns:
        # Filter out rows where best_eval didn't change to avoid clutter, or just plot line
        sns.lineplot(data=df, x="seen_steps", y="best_eval", linewidth=1.5, color="#E24A4A", linestyle="--", label="Peak Eval Score")

    plt.title("Agent Performance Over Training (10M Steps)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_return.png", dpi=300)
    print(f"Saved {OUTPUT_DIR}/training_return.png")
    plt.close()

    # Create the Hit Rate Plot
    plt.figure(figsize=(10, 6))
    
    # Apply rolling average to smooth the hit rate
    df['smooth_hit_rate'] = df['shot_hit_rate'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    
    sns.lineplot(data=df, x="seen_steps", y="smooth_hit_rate", linewidth=2.5, color="#50C878", label="Hit Rate (Smoothed)")
    
    plt.title("Shooting Accuracy Over Time", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Hit Rate (0.0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.0) # Hit rate is a percentage
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hit_rate.png", dpi=300)
    print(f"Saved {OUTPUT_DIR}/hit_rate.png")
    plt.close()

if __name__ == "__main__":
    main()