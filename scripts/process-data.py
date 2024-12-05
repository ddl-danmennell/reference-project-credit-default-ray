import os
import pandas as pd
from argparse import ArgumentParser
from flows import read_input, get_output_location

# Argument parser for allowing user to set inputs during location execution
parser = ArgumentParser(description='Data processing script.')
parser.add_argument('--merged_data', type=str, default='/mnt/code/outputs/merged_data', help='Path to the input data. Only used during local testing. Flow triggered jobs will use task inputs.')
parser.add_argument('--output_folder', type=str, default='/mnt/code/outputs', help='Path to output results. Only used during local testing. Flow triggered jobs will use task output directory.')
args = parser.parse_args()

# Read data input
merged_data = read_input(name='merged_data', args=args, is_file=True)

# Load data
df = pd.read_csv(merged_data) 

# Process data
print(df)
print('Processing the data ...')
df = df.drop('RandomColumn', axis=1)
print(df)

# Write output
output_location = get_output_location(name='processed_data', args=args)
df.to_csv(output_location, index=False)


