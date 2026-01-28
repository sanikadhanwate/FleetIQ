import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

#Sets a random state for reproducibility (match prior scripts' style)
RANDOM_STATE = 230



#Columns to remove (as provided)
DROP_COLS = [
    'Ambient Temperature',
    'Ambient Humidity',
    'Load Weight',
    'Driving Speed',
    'Distance Traveled',
    'Idle Time',
    'Route Roughness',
    'Timestamp',
]


df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv')

orig_shape = df.shape

#Normalize header spacing to align with provided column names (keep values unchanged)
df.columns = [c.strip() for c in df.columns]

#Drops irrelevant columns from the dataframe (ignore if any are absent)
to_drop = []
normalized = {c.replace('_', '').replace(' ', '').lower(): c for c in df.columns}
for col in DROP_COLS:
    key = col.replace('_', '').replace(' ', '').lower()
    if key in normalized:
        to_drop.append(normalized[key])

if to_drop:
    df.drop(columns=to_drop, inplace=True, errors='ignore')
print(f"Dropped columns: {to_drop if to_drop else 'None'}")

#Randomize the row order with a fixed seed and reset the index
df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# Saves the modified dataframe to a new CSV file
df.to_csv(str('EV_cleaned.csv'), index=False)



# Perform 80/10/10 split on the cleaned dataframe
EV_train, remaining_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
EV_val, EV_test = train_test_split(remaining_df, test_size=0.5, random_state=RANDOM_STATE)

# Build split file paths based on the cleaned output path



EV_train.to_csv()
EV_val.to_csv()
EV_test.to_csv()

print("Saved splits:")
print(f"  Train (80%): {EV_train.shape}")
print(f"  Validation (10%): {EV_val.shape}")
print(f"  Test (10%): {EV_test.shape}")
