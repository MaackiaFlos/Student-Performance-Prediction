import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'E:\Student Performance Prediction\final_df.csv')

# Select numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Plot heatmap of correlations among numeric features
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap for Numeric Features")
plt.show()

# Distribution of the target variable 'final_result'
if 'final_result' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="final_result")
    plt.title("Distribution of Final Results")
    plt.xlabel("Final Result")
    plt.ylabel("Count")
    plt.show()

# Pair plot for pairwise relationships among numeric features only
sns.pairplot(numeric_df, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle("Pairwise Relationships and Distributions", y=1.02)
plt.show()
