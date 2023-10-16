# Product-Demand-Prediction

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (assumes you have a CSV file with relevant data)
data = pd.read_csv('product_demand_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Use the trained model to make demand predictions
new_data = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
predicted_demand = model.predict(new_data)

# Output the predicted demand
print(f"Predicted Demand: {predicted_demand}")

