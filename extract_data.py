import os
import pandas as pd

file_dir = os.path.dirname(os.path.abspath(__file__))

training_file_name = 'epl-training.csv'

for root, dirs, files in os.walk(file_dir):
    for name in files:
        if name == training_file_name:
            data_file_path = os.path.join(root, name)

csv_data = pd.read_csv(data_file_path)
data_columns = list(csv_data)

for i in data_columns:
    if 'Unnamed' not in i:
        idx = data_columns.index(i)

csv_data_stripped = csv_data.iloc[:,:idx+1]
csv_data_formatted = pd.DataFrame(pd.to_datetime(csv_data_stripped.iloc[:,0]))
csv_data_formatted = pd.concat([csv_data_formatted,pd.get_dummies(csv_data_stripped.iloc[:,1:])],axis=1)

