import os
import json
import subprocess
import time

# --- CONFIGURATION ---

# 1. Set the exact results directory you want to check and add to.
#    This is the folder that contains your existing 'task...' subfolders.
RESULTS_DIR_TO_CHECK = r"results/Exp1_Baseline/20251207_22_56_39"

# 2. Set the path to the original, full dataset file.
ORIGINAL_DATASET_FILE = "data/WebVoyager_data.jsonl"

# 3. Define the command for running the agent.
#    This should be the same command you used originally.
#    The script will automatically modify '--test_file' and add '--specific_result_dir'.
BASE_COMMAND = [
    "python", "-u", "run.py",
    # "--test_file" will be added by the script
    "--api_key", "ollama",
    "--api_model", "qwen3-vl", # Make sure this is the model you want to use
    "--max_iter", "15",
    "--max_attached_imgs", "1",
    "--temperature", "0.1",
    "--fix_box_color",
    "--headless",
    "--window_width", "1024",
    "--window_height", "768",
    # "--specific_result_dir" will be added by the script
]


def find_existing_task_ids(results_dir):
    """Scans the results directory and returns a set of completed task IDs."""
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory '{results_dir}' not found. Assuming no tasks are complete.")
        return set()

    existing_ids = set()
    for folder_name in os.listdir(results_dir):
        # Check if it's a directory and follows the 'task...' naming convention
        if os.path.isdir(os.path.join(results_dir, folder_name)) and folder_name.startswith("task"):
            # Extract the ID from the folder name (e.g., "taskAllrecipes--0" -> "Allrecipes--0")
            task_id = folder_name[4:]
            existing_ids.add(task_id)
            
    print(f"Found {len(existing_ids)} existing task folders in '{results_dir}'.")
    return existing_ids


def create_missing_tasks_dataset(all_tasks_file, existing_ids):
    """Creates a temporary .jsonl file containing only the tasks that need to be run."""
    missing_tasks = []
    print(f"Reading full dataset from '{all_tasks_file}' to find missing tasks...")
    
    with open(all_tasks_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if not line.strip():
                continue
            try:
                task_data = json.loads(line)
                if task_data['id'] not in existing_ids:
                    missing_tasks.append(task_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping malformed line in original dataset: {e}")

    if not missing_tasks:
        return None, 0

    temp_dataset_path = "temp_missing_tasks.jsonl"
    with open(temp_dataset_path, 'w', encoding='utf-8') as f_out:
        for task in missing_tasks:
            f_out.write(json.dumps(task) + '\n')
            
    return temp_dataset_path, len(missing_tasks)


def main():
    """Main function to identify and run missing tasks."""
    print("--- SELECTIVE TASK RUNNER ---")
    
    # 1. Find which tasks are already done
    existing_ids = find_existing_task_ids(RESULTS_DIR_TO_CHECK)

    # 2. Create a temporary dataset with only the missing tasks
    temp_dataset_file, num_missing = create_missing_tasks_dataset(ORIGINAL_DATASET_FILE, existing_ids)

    if not temp_dataset_file:
        print("\nAll tasks are already complete. Nothing to do!")
        print("-----------------------------")
        return

    print(f"\nIdentified {num_missing} missing tasks that need to be run.")

    # 3. Construct the final command to run the agent
    final_cmd = list(BASE_COMMAND)
    final_cmd.extend([
        "--test_file", temp_dataset_file,
        "--specific_result_dir", RESULTS_DIR_TO_CHECK
    ])

    print("\n==================================================")
    print(f"STARTING EXECUTION for {num_missing} MISSING TASKS")
    print(f"Results will be saved in: {RESULTS_DIR_TO_CHECK}")
    print(f"Command: {' '.join(final_cmd)}")
    print("==================================================\n")

    start_time = time.time()
    
    try:
        # Run the command and wait for it to finish
        result = subprocess.run(final_cmd, check=False) # check=False to see errors in the output
        
        if result.returncode == 0:
            print(f"\nScript finished successfully.")
        else:
            print(f"\nScript failed with return code {result.returncode}.")
            
    except Exception as e:
        print(f"\nCritical error running subprocess: {e}")
    finally:
        # 4. Clean up the temporary file
        if os.path.exists(temp_dataset_file):
            os.remove(temp_dataset_file)
            print(f"Temporary dataset file '{temp_dataset_file}' has been removed.")
        
    duration = time.time() - start_time
    print(f"Total duration: {duration:.2f} seconds")
    print("--- SELECTIVE RUN COMPLETE ---")


if __name__ == "__main__":
    main()