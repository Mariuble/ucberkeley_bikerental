from sklearn.ensemble import GradientBoostingRegressor
import argparse
import pandas as pd
import xgboost as xgb
import joblib

# Arguments for paths
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/train.csv')
parser.add_argument('--model_path', type=str, default='models/gbr.joblib')
args = parser.parse_args()

# Best model parameters from grid search fine-tuning
random_state = 99
learning_rate = 0.1
max_depth = 9
min_samples_leaf = 9
n_estimators = 100

# Load training data
train_data = pd.read_csv(args.data_path)
X_train, y_train = train_data.drop('cnt', axis=1), train_data['cnt']

# Set up model
model = GradientBoostingRegressor(
    random_state=random_state, 
    learning_rate=learning_rate, 
    max_depth=max_depth, 
    min_samples_leaf=min_samples_leaf, 
    n_estimators=n_estimators
)

# Train model
model.fit(X_train, y_train)

# Save model
# model.save_model(args.model_path) # Doesn't work
joblib.dump(model, args.model_path)

print("Model saved to", args.model_path)