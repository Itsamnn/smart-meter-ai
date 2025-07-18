import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error  # To measure accuracy
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('data/AEP_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

def prepare_features(df):
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['is_weekend'] = df['Datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year
    return df

# --- NEW: We will split the data into a training set and a testing set ---
def split_data(df):
    # We will train on all data before the year 2015
    # And test on all data from 2015 onwards
    train_data = df[df['Datetime'] < '2015-01-01']
    test_data = df[df['Datetime'] >= '2015-01-01']
    
    print(f"\nTraining data from {train_data['Datetime'].min()} to {train_data['Datetime'].max()}")
    print(f"Testing data from {test_data['Datetime'].min()} to {test_data['Datetime'].max()}")
    
    # Define features and target
    features = ['hour', 'dayofweek', 'is_weekend', 'month', 'year']
    target = 'AEP_MW'
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    # Replace LinearRegression with a more powerful model
    # n_estimators=10 means we're creating a 'committee' of 10 decision trees
    # n_jobs=-1 tells it to use all your computer's processors to speed up training
    # random_state=42 ensures you get the same result every time you run it
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Mean Absolute Error: {mae:.2f} MW")
    return predictions

def make_predictions(model, X):
    print("\nUsing the trained model to make predictions...")
    predictions = model.predict(X)
    print("Predictions are ready.")
    return predictions

def plot_predictions(df, y, predictions):
    print("Generating comparison plot...")
    plt.figure(figsize=(15, 7))
    
    # Plot actual vs predicted
    plt.plot(df['Datetime'], y, label='Actual Energy Usage', color='blue', linewidth=2)
    plt.plot(df['Datetime'], predictions, label='AI Prediction', color='red', linestyle='--')
    
    plt.title('AI Energy Prediction vs. Actual Usage')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption (MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
df = load_data()
df = prepare_features(df)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
print(f"Average MW: {df['AEP_MW'].mean():.0f}")

# Split data into training and testing sets
X_train, y_train, X_test, y_test = split_data(df)

# Train the model
model = train_model(X_train, y_train)
print("Model trained successfully!")

# Evaluate model on test data
test_predictions = evaluate_model(model, X_test, y_test)

# Create a subset of test data for visualization
test_df = df[df['Datetime'] >= '2015-01-01'].head(1000)
test_pred_subset = test_predictions[:1000]

# Plot predictions vs actual
plot_predictions(test_df, test_df['AEP_MW'], test_pred_subset)
print("\nProgram finished. You can close the plot window to exit.") 