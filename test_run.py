from pipeline import run_pipeline
from config import AppConfig

cfg = AppConfig(
    file_path="data/Greece.xlsx",
    mode="data-driven",
    use_clustering=True,
    n_clusters=4,
    portfolio_size=7,
)

results = run_pipeline(cfg)

print("\n=== DONE ===")
print("Portfolio:")
print(results["portfolio_df"].head(20))

print("\nTop ranked:")
cols = [c for c in ["CompID", "TOTAL_SCORE", "cluster"] if c in results["ranked_df"].columns]
print(results["ranked_df"][cols].head(20))