import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Define the feature names with units for clarity
features = [
    'Engine displacement (cu.in.)',
    'Acceleration (0-60 mph in sec)',
    'Number of cylinders',
    'Horsepower (hp)',
    'Weight (pounds)'
]

# Function to collect car data from the user
def get_user_data():
    """Collect training data for cars from user input."""
    print("Enter car data. Type 'done' when finished.")
    data = []
    while True:
        entry = {}
        for feature in features:
            while True:
                if feature == 'Weight (pounds)':
                    value = input(f"Enter {feature} (e.g., 3500) or 'done': ")
                else:
                    value = input(f"Enter {feature} (e.g., 305 for Engine displacement) or 'done': ")
                if value.lower() == 'done':
                    if not data:
                        print("No data entered.")
                        return pd.DataFrame()
                    return pd.DataFrame(data)
                try:
                    entry[feature] = float(value)
                    break
                except ValueError:
                    print(f"Please enter a valid number for {feature} or 'done'.")
        while True:
            mpg = input("Enter MPG (e.g., 30): ")
            try:
                entry['MPG'] = float(mpg)
                data.append(entry)
                break
            except ValueError:
                print("Please enter a valid number for MPG.")
    return pd.DataFrame(data)

# Function to train the model with multiple features
def train_model(data):
    """Train a RidgeCV regression model on the collected data with standardized features."""
    X = data[features].values  # Feature matrix
    y = data['MPG'].values     # MPG values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use RidgeCV with a wide range of alphas, including very small values for small datasets
    model = RidgeCV(alphas=np.logspace(-6, 3, 10))
    model.fit(X_scaled, y)
    
    return model, scaler, model.coef_, model.intercept_, model.alpha_

# Function to predict MPG for a new car
def predict_mpg(model, scaler):
    """Predict MPG for new cars based on user input with standardized features."""
    print("\nNow, predict MPG for new cars by entering their features.")
    while True:
        entry = []
        for feature in features:
            while True:
                if feature == 'Weight (pounds)':
                    value = input(f"Enter {feature} for prediction (e.g., 3600) or 'exit': ")
                else:
                    value = input(f"Enter {feature} for prediction (e.g., 145 for Horsepower) or 'exit': ")
                if value.lower() == 'exit':
                    print("Exiting prediction mode.")
                    return
                try:
                    entry.append(float(value))
                    break
                except ValueError:
                    print(f"Please enter a valid number for {feature} or 'exit'.")
        # Standardize the new entry using the same scaler
        entry_scaled = scaler.transform([entry])
        prediction = model.predict(entry_scaled)[0]
        print(f"Predicted MPG: {prediction:.2f}")

# Main program
def main():
    """Main function to run the Car MPG Predictor."""
    print("Welcome to the Car MPG Predictor!")
    # Get data
    data = get_user_data()
    
    if len(data) < 2:
        print("Need at least 2 data points to train the model.")
        return
    
    # Train the model
    model, scaler, weights, bias, selected_alpha = train_model(data)
    print("\nModel trained successfully!")
    print(f"Selected alpha: {selected_alpha}")
    print(f"Bias (y-intercept): {bias:.2f}")
    for feature, weight in zip(features, weights):
        print(f"Weight for {feature}: {weight:.4f}")
    
    # Make predictions
    predict_mpg(model, scaler)

if __name__ == "__main__":
    main()