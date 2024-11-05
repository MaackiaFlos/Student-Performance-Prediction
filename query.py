import pandas as pd

# Load the dataset
df = pd.read_csv(r'E:\Student Performance Prediction\final_df.csv')

# Display basic statistics for numerical columns
print("Dataset Description:")
print(df.describe())

# Display data information for numerical columns
print("Dataset Information:")
print(df.info())