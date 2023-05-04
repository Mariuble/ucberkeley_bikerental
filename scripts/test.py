import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/test.csv')
parser.add_argument('--model_path', type=str, default='models/gbr.joblib')
parser.add_argument('--output_path', type=str, default='predictions/predictions.csv')
args = parser.parse_args()

# Load test data
test_data = pd.read_csv(args.data_path)
X_test = test_data.drop('cnt', axis=1)
y_test = test_data['cnt']

# Load model
model = joblib.load(args.model_path)

# Make predictions
predictions = model.predict(X_test)

# Save predictions
np.savetxt(args.output_path, predictions, delimiter=',')
print('Predictions saved to', args.output_path)

print('Evaluating predictions...')
r2 = r2_score(y_test, predictions)
print('Test set R-squared value: {:.4f}'.format(r2))
