import pandas as pd
import os

# Configuration
DATASET_CSV = "master_dataset.csv"

# The labels your auto_labeler uses
LABELS = [
    "HOMEPAGE", 
    "LISTING_PAGE", 
    "DETAIL_PAGE", 
    "INPUT_FORM", 
    "LOGIN_SIGNUP", 
    "CAPTCHA_SECURITY", 
    "ERROR_404", 
    "POPUP_OVERLAY",
    "OTHER",
    "MODEL_ERROR", # Safety label if classification fails
    "IMAGE_READ_ERROR" # Safety label
]

def analyze_dataset():
    if not os.path.exists(DATASET_CSV):
        print(f"Error: Dataset file '{DATASET_CSV}' not found. Please run the auto_labeler first.")
        return

    print("--- DATASET SANITY CHECK ---")

    # Load the dataset with pandas
    try:
        df = pd.read_csv(DATASET_CSV)
    except Exception as e:
        print(f"Could not read CSV file: {e}")
        return
        
    if df.empty:
        print("Dataset is empty. No data to analyze.")
        return

    # --- 1. Analyze Task Success/Failure ---
    print("\n[1] Task Status Analysis")
    
    # We need to count unique tasks, not every row
    # Group by task_id and take the first status entry for each
    task_statuses = df.groupby('task_id')['task_status'].first()
    
    successful_tasks = (task_statuses == 'Success').sum()
    failed_tasks = (task_statuses != 'Success').sum()
    total_tasks = len(task_statuses)

    print(f"  - Total unique tasks processed: {total_tasks}")
    print(f"  - Successful tasks: {successful_tasks}")
    print(f"  - Failed tasks: {failed_tasks}")
    
    if total_tasks > 0:
        success_rate = (successful_tasks / total_tasks) * 100
        print(f"  - Agent Success Rate: {success_rate:.2f}%")


    # --- 2. Analyze Image Classes (Overall) ---
    print("\n[2] Overall Image Class Distribution (All Tasks)")
    
    class_counts = df['label'].value_counts().reindex(LABELS, fill_value=0)
    total_images = len(df)
    
    print(f"  - Total images labeled: {total_images}")
    for label, count in class_counts.items():
        if total_images > 0:
            percentage = (count / total_images) * 100
            print(f"    - {label:<20}: {count:<5} images ({percentage:.2f}%)")
        else:
            print(f"    - {label:<20}: 0 images")

    # --- 3. Analyze Image Classes (Successful Tasks Only) ---
    print("\n[3] Image Class Distribution (Successful Tasks Only)")
    
    # Filter the dataframe to only include rows from successful tasks
    successful_df = df[df['task_status'] == 'Success']
    
    if not successful_df.empty:
        success_class_counts = successful_df['label'].value_counts().reindex(LABELS, fill_value=0)
        total_success_images = len(successful_df)

        print(f"  - Total images from successful tasks: {total_success_images}")
        for label, count in success_class_counts.items():
            percentage = (count / total_success_images) * 100
            print(f"    - {label:<20}: {count:<5} images ({percentage:.2f}%)")
    else:
        print("  - No successful tasks found to analyze.")
        
    print("\n--- SANITY CHECK COMPLETE ---")


if __name__ == "__main__":
    analyze_dataset()