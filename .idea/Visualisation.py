import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('.idea/TelcoCustomerChurn.csv')

# Display basic info
print(df.head())
print(df.info())

# Create a scatter plot for numerical columns
# Example: Monthly Charges vs Tenure
plt.figure(figsize=(10, 6))
plt.scatter(df['tenure'], df['MonthlyCharges'], alpha=0.5)

plt.xlabel('Tenure (months)')
plt.ylabel('Monthly Charges ($)')
plt.title('Customer Tenure vs Monthly Charges')
plt.grid(True)
plt.show()

# Create a scatter plot for numerical columns with different colors based on a condition
plt.figure(figsize=(10, 6))
colors = np.where(df['MonthlyCharges'] > df['MonthlyCharges'].median(), 'r', 'b')  # Red for above median, blue for below

# Create scatter plots with labels for legend
above_median = df['MonthlyCharges'] > df['MonthlyCharges'].median()
plt.scatter(df[above_median]['tenure'], df[above_median]['MonthlyCharges'], alpha=0.5, c='r', label='Above Median')
plt.scatter(df[~above_median]['tenure'], df[~above_median]['MonthlyCharges'], alpha=0.5, c='b', label='Below Median')

plt.xlabel('Tenure (months)')
plt.ylabel('Monthly Charges ($)')
plt.title('Customer Tenure vs Monthly Charges')
plt.legend()
plt.grid(True)
plt.show()
