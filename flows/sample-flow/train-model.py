import os
import pickle
import pandas as pd
from argparse import ArgumentParser
from flows import read_input, get_output_location
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Argument parser for allowing user to set inputs during location execution
parser = ArgumentParser(description='Model training script.')
parser.add_argument('--processed_data', type=str, default='/mnt/code/outputs/processed_data', help='Path to the input data. Only used during local testing. Flow triggered jobs will use task inputs.')
parser.add_argument('--num_estimators', type=str, default=100, help='The number of trees in the forest. Only used during local testing. Flow triggered jobs will use task inputs.')
parser.add_argument('--output_folder', type=str, default='/mnt/code/outputs', help='Path to output results. Only used during local testing. Flow triggered jobs will use task output directory.')
args = parser.parse_args()

# Read inputs
processed_data = read_input(name='processed_data', args=args, is_file=True)
num_estimators = read_input(name='num_estimators', args=args)

# Load data
df = pd.read_csv(processed_data) 

# Separate features and labels
X = df.drop(columns=['Species'])
y = df['Species']

# Encode labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print('\nTraining set size:', X_train.shape)
print('Testing set size:', X_test.shape)

# Train a model (e.g., Random Forest Classifier)
model = RandomForestClassifier(random_state=42, n_estimators=int(num_estimators))
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('\nModel Accuracy on Test Set:', accuracy)
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Write the mode as output
output_location = get_output_location(name='model', args=args)
with open(output_location, 'wb') as file:
    pickle.dump(model, file)

