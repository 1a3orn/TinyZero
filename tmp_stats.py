import json
from collections import Counter
import pandas as pd

# Read the JSON file
with open('data_sft_0_5000.json', 'r') as f:
    data = json.load(f)

# Filter to correct solutions only
correct_solutions = [item for item in data if item.get('is_correct', False)]

# Count lengths of numbers_to_use and numbers_available
numbers_to_use_lengths = Counter(len(item['numbers_to_use']) for item in correct_solutions)
numbers_available_lengths = Counter(len(item['numbers_available']) for item in correct_solutions)

# Convert to pandas DataFrames for nice display
df_to_use = pd.DataFrame.from_dict(numbers_to_use_lengths, orient='index', columns=['Count'])
df_to_use.index.name = 'Length'
df_to_use.name = 'Numbers To Use'

df_available = pd.DataFrame.from_dict(numbers_available_lengths, orient='index', columns=['Count'])
df_available.index.name = 'Length'
df_available.name = 'Numbers Available'

print("Statistics for numbers_to_use:")
print(df_to_use)
print("\nStatistics for numbers_available:")
print(df_available)