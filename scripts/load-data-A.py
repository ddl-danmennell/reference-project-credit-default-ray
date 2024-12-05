import os
import pandas as pd
from argparse import ArgumentParser
from flows import read_input, get_output_location

# Argument parser for allowing user to set inputs during location execution
parser = ArgumentParser(description='Data preparation script.')
parser.add_argument('--data_path', type=str, default='/mnt/code/data/datasetA.csv', help='Path to the input data. Only used during local testing. Flow triggered jobs will use task inputs.')
parser.add_argument('--output_folder', type=str, default='/mnt/code/outputs', help='Path to output results. Only used during local testing. Flow triggered jobs will use task output directory.')
args = parser.parse_args()

# Read data input
data_path = read_input(name='data_path', args=args)

# Load dataset
print('Loading in dataset A...')
df = pd.read_csv(data_path) 

# Write output
output_location = get_output_location(name='datasetA', args=args)
df.to_csv(output_location, index=False)
