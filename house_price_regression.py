import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#  Load dataset
df = pd.read_csv(r"C:\Users\VAIBHAV\OneDrive\Desktop\house_price_prediction\data\train.csv")

# Create 'Bathrooms' feature
df['Bathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

# Select features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'Bathrooms']]  # Features: square footage, bedrooms, bathrooms
y = df['SalePrice']  # Target: house price

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Optional show actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nFirst 10 predictions:\n", comparison.head(10))

# Scatter plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
