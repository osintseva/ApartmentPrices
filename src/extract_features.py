import pandas as pd
import numpy as np
from gpt_text_to_properties import text_to_json
import json
import time
import os

def save_checkpoint(df, results, checkpoint_num):
    df_results = pd.DataFrame(results)
    if 'ID' not in df_results.columns and 'id' in df_results.columns:
        df_results['ID'] = df_results['id']
        df_results = df_results.drop('id', axis=1)
    df_results['ID'] = df_results['ID'].astype(df['ID'].dtype)
    df_merged = pd.merge(df.iloc[:len(results)], df_results, on='ID', how='left')
    df_merged.to_csv(f"checkpoint_{checkpoint_num}.csv", index=False)
    print(f"\nCheckpoint {checkpoint_num} saved.")

try:
    # Load the original dataframe
    df = pd.read_csv("df_cleaned.csv")
    print(f"Loaded {len(df)} rows from df_cleaned.csv")

    # Initialize a list to store results
    results = []
    checkpoint_interval = 1000
    last_checkpoint = 0

    # Process each row in the dataframe
    for index, row in df.iterrows():
        try:
            description = row['Description']
            id = row['ID'] if 'ID' in df.columns else index
            
            json_result = text_to_json(description, id)
            if json_result:
                results.append(json.loads(json_result))
                print(f"\nProcessed ID {id}:\n{json_result}")
            else:
                print(f"\nFailed to process ID {id}")
            
            # Save checkpoint every 1000 entries
            if (index + 1) % checkpoint_interval == 0:
                save_checkpoint(df, results, (index + 1) // checkpoint_interval)
                last_checkpoint = index + 1
            
            # Add a small delay to avoid hitting API rate limits
            #time.sleep(1)
        
        except Exception as e:
            print(f"Error processing row {index} (ID: {id}): {str(e)}")
            continue  # Skip to the next row if there's an error
        
        # Uncomment the following lines if you want to process only the first 10 rows
        # if index == 9:
        #     break

    # Save final results
    df_results = pd.DataFrame(results)

    # Ensure 'ID' column is present and has the correct data type
    if 'ID' not in df_results.columns and 'id' in df_results.columns:
        df_results['ID'] = df_results['id']
        df_results = df_results.drop('id', axis=1)

    # Convert 'ID' to the same data type as in the original dataframe
    df_results['ID'] = df_results['ID'].astype(df['ID'].dtype)

    # Merge the new features with the original dataframe
    df_merged = pd.merge(df, df_results, on='ID', how='left')

    # Print the first few rows of the merged dataframe
    print("\nMerged Dataframe:")
    print(df_merged.head())

    # Save the merged dataframe to a new CSV file
    df_merged.to_csv("df_with_new_features.csv", index=False)
    print("\nMerged dataframe saved as 'df_with_new_features.csv'")

    # Remove checkpoint files
    for i in range(1, (last_checkpoint // checkpoint_interval) + 1):
        os.remove(f"checkpoint_{i}.csv")
    print("Checkpoint files removed.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    if results:
        print("Saving progress before exit...")
        save_checkpoint(df, results, "final")
    raise  # Re-raise the exception for debugging purposes