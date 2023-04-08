import pandas as pd

# Load the data from the JSON file
with open('iot_sensors_data.json', 'r') as f:
    iot_sensors_data = json.load(f)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(iot_sensors_data['data'])

# Print some basic statistics about the numerical variables
print(df.describe())

# Count the number of unique values for each variable
for column in df.columns:
    print(f"Number of unique values for {column}: {len(df[column].unique())}")
