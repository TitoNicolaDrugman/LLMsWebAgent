import os
import json
import shutil
import glob
import subprocess

# --- CONFIGURATION ---
# The folder where your crashed results are
RESULTS_BASE_DIR = r"results\Exp1_Baseline\20251207_22_56_39"
# The original full dataset file
ORIGINAL_DATASET = "data/WebVoyager_data.jsonl"
# Temporary file to store the list of tasks to re-run
TEMP_DATASET = "temp_resume_tasks.jsonl"

def main():
    print(f"Scanning {RESULTS_BASE_DIR} for incomplete tasks...")
    
    tasks_to_rerun = []
    
    # 1. Scan the results folder
    if not os.path.exists(RESULTS_BASE_DIR):
        print("Error: Results directory not found.")
        return

    # Iterate over subdirectories (e.g., taskAllrecipes--0)
    for task_folder_name in os.listdir(RESULTS_BASE_DIR):
        task_path = os.path.join(RESULTS_BASE_DIR, task_folder_name)
        
        if not os.path.isdir(task_path):
            continue
            
        # Check if it starts with 'task'
        if not task_folder_name.startswith("task"):
            continue

        # Count screenshots
        png_files = glob.glob(os.path.join(task_path, "*.png"))
        num_pngs = len(png_files)
        
        # logic: if <= 2 screenshots, consider it crashed/cold start fail
        if num_pngs <= 2:
            print(f" -> Found incomplete task: {task_folder_name} ({num_pngs} images). Preparing to re-run.")
            
            # Extract ID from folder name (remove "task" prefix)
            # Folder: taskAllrecipes--0  -> ID: Allrecipes--0
            task_id = task_folder_name[4:] 
            tasks_to_rerun.append(task_id)
            
            # CLEANUP: Delete files inside this folder so we start fresh
            # We keep the folder itself, but empty it.
            for file in os.listdir(task_path):
                file_path = os.path.join(task_path, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f" -> Skipping {task_folder_name} ({num_pngs} images) - seems okay.")

    print(f"\nTotal tasks to re-run: {len(tasks_to_rerun)}")
    
    if len(tasks_to_rerun) == 0:
        print("No tasks need re-running. Exiting.")
        return

    # 2. Create the temporary dataset
    print(f"Creating temporary dataset: {TEMP_DATASET}...")
    found_count = 0
    with open(TEMP_DATASET, 'w', encoding='utf-8') as outfile:
        with open(ORIGINAL_DATASET, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if data['id'] in tasks_to_rerun:
                        outfile.write(json.dumps(data) + "\n")
                        found_count += 1
                except:
                    pass
    
    print(f"Added {found_count} tasks to temporary dataset.")

    # 3. Construct the run command
    # We use the flags you used before, but point to the temp file and specific result dir
    cmd = [
        "python", "-u", "run.py",
        "--test_file", TEMP_DATASET,
        "--api_key", "ollama",
        "--api_model", "qwen3-vl",
        "--max_iter", "15",
        "--max_attached_imgs", "1",
        "--temperature", "0.1",
        "--fix_box_color",
        "--headless",
        "--window_width", "1024",
        "--window_height", "768",
        # CRITICAL: Force the existing directory
        "--specific_result_dir", RESULTS_BASE_DIR 
    ]
    
    print("\nStarting execution...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("\nResume run completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError running script: {e}")
    finally:
        # Optional: Cleanup temp file
        if os.path.exists(TEMP_DATASET):
            os.remove(TEMP_DATASET)
            print("Temporary dataset removed.")

if __name__ == "__main__":
    main()