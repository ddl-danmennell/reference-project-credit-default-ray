import os
import pandas as pd
from argparse import ArgumentParser
from flows import read_input, get_output_location

# Argument parser for allowing user to set inputs during location execution
parser = ArgumentParser(description='Model training script.')
parser.add_argument('--datasetA', type=str, default='/mnt/code/outputs/datasetA', help='Path to the input data. Only used during local testing. Flow triggered jobs will use task inputs.')
parser.add_argument('--datasetB', type=str, default='/mnt/code/outputs/datasetB', help='Path to the input data. Only used during local testing. Flow triggered jobs will use task inputs.')
parser.add_argument('--output_folder', type=str, default='/mnt/code/outputs', help='Path to output results. Only used during local testing. Flow triggered jobs will use task output directory.')
args = parser.parse_args()

# Read inputs
datasetA = read_input(name='datasetA', args=args, is_file=True)
datasetB = read_input(name='datasetB', args=args, is_file=True)

# Load data
a = pd.read_csv(datasetA, index_col='Id') 
b = pd.read_csv(datasetB, index_col='Id') 

# Merge data
print('Merging data...')
merged = pd.concat([a, a], axis=0).reset_index(drop=True)
print(merged)

# Write output
output_location = get_output_location(name='merged_data', args=args)
merged.to_csv(output_location, index=False)
