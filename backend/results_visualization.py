import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# LOAD RESULTS
# ==========================================================

df = pd.read_csv("model_results.csv")

# ==========================================================
# MODEL ACCURACY COMPARISON
# ==========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="Model",
    y="Accuracy",
    hue="Optimization"
)

plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("ml_model_comparison.png")
plt.close()

# ==========================================================
# OPTIMIZATION TECHNIQUE COMPARISON
# ==========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="Optimization",
    y="Accuracy"
)

plt.title("Optimization Technique Comparison")

plt.tight_layout()
plt.savefig("optimization_comparison.png")
plt.close()

# ==========================================================
# MODEL PERFORMANCE (F1 SCORE)
# ==========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="Model",
    y="F1",
    hue="Optimization"
)

plt.title("Model Performance (F1 Score)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("model_performance.png")
plt.close()

# ==========================================================
# HEATMAP
# ==========================================================

pivot = df.pivot_table(
    index="Model",
    columns="Optimization",
    values="Accuracy"
)

plt.figure(figsize=(10,6))

sns.heatmap(
    pivot,
    annot=True,
    cmap="YlGnBu"
)

plt.title("Accuracy Heatmap")

plt.tight_layout()
plt.savefig("accuracy_heatmap.png")
plt.close()

print("Graphs updated successfully.")