
import sys
from pathlib import Path
import pandas as pd
import json

# Ensure src is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.empirical.hypothesis_test import run_full_hypothesis_test

def main():
    input_path = "outputs/annotations/v3/time_series_10m_plus.csv"
    print(f"Loading {input_path}...")
    try:
        df = pd.read_csv(input_path)
        if "time_window" in df.columns:
            df["time_window"] = pd.to_datetime(df["time_window"])
    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return

    print(f"Running hypothesis tests on {len(df)} time windows...")
    results = run_full_hypothesis_test(df, topic_name="Long-COVID (Merged)")

    # Print relevant results
    print("\n=== Hypothesis Test Results ===")
    
    # H1: Activity vs Jump
    print("\n[H1] High Activity -> Jumps:")
    h1 = results["hypothesis_support"]["H1_high_a_high_jump"]
    print(f"  Supported: {h1}")
    metrics = results["jump_metrics"]
    print(f"  Jump Score: {metrics['jump_score']:.3f} (Threshold > 0.3)")
    print(f"  Max dP/dt: {metrics['dP_dt_max']:.3f}")
    
    # H2 & H3: r_proxy interaction
    print("\n[H2 & H3] Media Control (r_proxy) Effects:")
    effects = results["parameter_effects"]
    if "error" not in effects:
        h2 = results["hypothesis_support"]["H2_high_r_high_volatility"]
        h3 = results["hypothesis_support"]["H3_interaction_effect"]
        
        r_vol = effects.get("r_proxy_vs_volatility", {})
        print(f"  [H2] r_proxy vs Volatility Correlation: {r_vol.get('correlation', 0):.3f} (p={r_vol.get('p_value', 1):.3f})")
        print(f"  Supported: {h2}")
        
        inter = effects.get("interaction_effect", {})
        print(f"  [H3] Interaction (High r/High a vs Low r/Low a):")
        if inter:
            print(f"    High-High Volatility: {inter.get('high_r_high_a_volatility', 0):.3f}")
            print(f"    Low-Low Volatility: {inter.get('low_r_low_a_volatility', 0):.3f}")
            print(f"    Significant: {inter.get('significant', False)}")
        print(f"  Supported: {h3}")
    else:
        print(f"  Error in interaction test: {effects.get('error')}")

    # H4: Critical Slowing Down
    print("\n[H4] Critical Slowing Down (CSD):")
    h4 = results["hypothesis_support"]["H4_csd_before_jump"]
    csd = results["csd_signals"]
    print(f"  Supported: {h4}")
    print(f"  AC1 Trend: {csd['ac1_trend']:.4f} (Should be > 0)")
    print(f"  Variance Trend: {csd['var_trend']:.4f} (Should be > 0)")

    # Save full results
    out_json = "outputs/hypothesis_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_json}")

if __name__ == "__main__":
    main()
