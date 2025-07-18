import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def load_data():
    df = pd.read_csv('data/AEP_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

def prepare_features(df):
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)  # Saturday=5, Sunday=6
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year
    return df

def show_data_info(df):
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"Average MW: {df['AEP_MW'].mean():.0f}")

def plot_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Datetime'][:168], df['AEP_MW'][:168], color='steelblue')
    plt.title('AEP Power Consumption (First Week)')
    plt.xlabel('Date')
    plt.ylabel('Power (MW)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train_model(df):
    # Add 'is_weekend' to our list of clues
    features = ['hour', 'dayofweek', 'is_weekend', 'month', 'year']
    X = df[features]
    y = df['AEP_MW']
    
    # Replace LinearRegression with a more powerful model
    # n_estimators=10 means we're creating a 'committee' of 10 decision trees
    # n_jobs=-1 tells it to use all your computer's processors to speed up training
    # random_state=42 ensures you get the same result every time you run it
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    
    model.fit(X, y)
    return model, X, y

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
show_data_info(df)
plot_data(df)
model, X, y = train_model(df)
print("Model trained successfully!")

predictions = make_predictions(model, X)
plot_predictions(df, y, predictions)
print("\nProgram finished. You can close the plot window to exit.") 