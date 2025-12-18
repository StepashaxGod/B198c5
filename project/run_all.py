from data_prep import load_wine_dataset, make_dirty_dataset
from manual_pipeline import clean_manual_fixed
from adaptive_pipeline import get_strategies
from metrics_utils import evaluate_model


def main():
    target = "quality"

    data = load_wine_dataset("winequality-red.csv")
    if target not in data.columns:
        raise KeyError(
            f"Target column '{target}' not found. Available columns: {data.columns.tolist()}"
        )
    X = data.drop(columns=[target])
    y = data[target]

    print("Columns:", data.columns.tolist())
    print(f"Original shape: X={X.shape}, y={y.shape}")

    data_dirty = make_dirty_dataset(
        data,
        target=target,
        missing_rate=0.05,
        noise_std=0.02,
        n_duplicates=50,
        outlier_rate=0.03,
        outlier_multiplier=6.0,
        seed=42,
    )

    X_dirty = data_dirty.drop(columns=[target])
    y_dirty = data_dirty[target]
    print(f"After dirtying: X_dirty={X_dirty.shape}, y_dirty={y_dirty.shape}")

    cleaned_manual = clean_manual_fixed(data_dirty, target=target)
    rmse_m, mae_m, r2_m, t_m = evaluate_model(cleaned_manual, target=target)

    print("\nManual Cleaning Pipeline (fixed)")
    print(f"Rows: {len(data_dirty)} -> {len(cleaned_manual)}")
    print(f"RMSE: {rmse_m:.3f}")
    print(f"MAE:  {mae_m:.3f}")
    print(f"R²:   {r2_m:.3f}")
    print(f"Time: {t_m:.2f}s")

    strategies = get_strategies(target=target)
    results = {}
    for name, fn in strategies.items():
        cleaned = fn(data_dirty.copy())
        rmse, mae, r2, t = evaluate_model(cleaned, target=target)
        results[name] = (len(cleaned), rmse, mae, r2, t)

    best_name = min(results.items(), key=lambda kv: kv[1][1])[0]

    print("\nAdaptive Pipeline (evaluated strategies)")
    for method, (rows_after, rmse, mae, r2, t) in results.items():
        print(f"{method:>20}: Rows={rows_after}, RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}, Time={t:.2f}s")

    print(f"\nBest method by RMSE: {best_name}")

    rows_b, rmse_b, mae_b, r2_b, t_b = results[best_name]
    print("\nSummary Comparison")
    print(f"Manual Fixed:  RMSE={rmse_m:.3f}, MAE={mae_m:.3f}, R²={r2_m:.3f}, Time={t_m:.2f}s")
    print(f"Best Adaptive: RMSE={rmse_b:.3f}, MAE={mae_b:.3f}, R²={r2_b:.3f}, Time={t_b:.2f}s, Rows={rows_b}")


if __name__ == "__main__":
    main()

