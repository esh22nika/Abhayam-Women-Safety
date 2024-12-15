import pandas as pd

# Load the data from CSV
df = pd.read_csv('violence_log.csv')

# Group by 'location' and 'Action Detected' and count rows
grouped_df = df.groupby(['Location', 'Action Detected']).size().reset_index(name='Count')

# Filter rows where 'Action Detected' is "Violence against a woman" and Count > 5
filtered_df = grouped_df[(grouped_df['Action Detected'] == 'Violence against a woman') & (grouped_df['Count'] > 5)]

# Extract only the desired columns
filtered_df = filtered_df[['Location', 'Action Detected', 'Count']]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('hotspot.csv', index=False)

print("Filtered data saved to 'hotspot.csv' with selected columns.")
