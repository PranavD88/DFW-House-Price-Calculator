import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Load and preprocess the dataset
file_path = r"C:\Users\WinterSongMC\Desktop\DFW real Estate Data.csv"
data = pd.read_csv(file_path)
data = data.dropna(subset=['BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT', 'PRICE', 'CITY'])
data = data[data['PRICE'] <= 1000000]
data = pd.get_dummies(data, columns=['CITY'], drop_first=True)
features = ['BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT'] + [col for col in data.columns if 'CITY_' in col]
X, y = data[features], data['PRICE']

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Visualizations
importance = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importance[np.argsort(importance)[-10:]], y=np.array(features)[np.argsort(importance)[-10:]])
plt.title('Top 10 Feature Importance'.title())
plt.xlabel('Importance'.title())
plt.ylabel('Features'.title())
plt.show()
correlation = np.corrcoef(y_test, y_pred)[0, 1]
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.title('Actual Vs. Predicted Prices With Linear Regression Line'.title())
plt.text(x=0.05 * max(y_test), y=0.95 * max(y_pred), s=f"r = {correlation:.2f}", fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel('Actual Prices (In Dollars)'.title())
plt.ylabel('Predicted Prices (In Dollars)'.title())
plt.show()

# Function to get valid numeric input
def get_numeric_input(prompt, input_type=int, condition=lambda x: True, error_message="Invalid input. Please try again."):
    while True:
        try:
            value = input_type(input(prompt))
            if condition(value): return value
            else: print(error_message)
        except ValueError: print(error_message)

# Predict a price based on user input
print("\nEnter the details for a new house to predict its price:")
user_input = {
    'BEDS': get_numeric_input("Number of bedrooms (BEDS): ", int, lambda x: x > 0),
    'BATHS': get_numeric_input("Number of bathrooms (BATHS): ", int, lambda x: x > 0),
    'SQUARE FEET': get_numeric_input("Square footage (SQUARE FEET): ", int, lambda x: x > 0),
    'LOT SIZE': get_numeric_input("Lot size (LOT SIZE, in square feet): ", int, lambda x: x > 0),
    'YEAR BUILT': get_numeric_input("Year built (YEAR BUILT): ", int, lambda x: 1800 <= x <= 2025)
}
cities = [col.replace('CITY_', '') for col in features if 'CITY_' in col]
while True:
    city = input("City (CITY): ").strip().title()
    if city not in cities: print(f"Sorry, the data for the city '{city}' does not exist in this program. Please try again.")
    else: break
for col in X_train.columns:
    if col.startswith("CITY_"):
        user_input[col] = 1 if f"CITY_{city}" == col else 0
    elif col not in user_input:
        user_input[col] = 0

user_input_df = pd.DataFrame([user_input], columns=X_train.columns)

# Predict the price
prediction = model.predict(user_input_df)[0]
print(f"\nPredicted Price: ${prediction:,.2f}")
