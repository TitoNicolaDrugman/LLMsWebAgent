import os
import subprocess
import time

# Define your experiments here
# You can vary prompts, models, or flags
experiments = [
    {
        "name": "Exp1_Baseline_Qwen",
        "cmd": [
            "python", "-u", "run.py",
            "--test_file", "./data/WebVoyager_data.jsonl",
            "--api_key", "ollama",
            "--api_model", "qwen3-vl",
            "--max_iter", "15",
            "--max_attached_imgs", "1",
            "--temperature", "0.1",
            "--fix_box_color",
            "--headless", 
            "--window_width", "1024",
            "--window_height", "768",
            "--output_dir", "results/Exp1_Baseline"
        ]
    },
    {
        "name": "Exp2_NoBoxColor_Qwen",
        # Example: removing --fix_box_color to see if random colors help/hurt
        "cmd": [
            "python", "-u", "run.py",
            "--test_file", "./data/WebVoyager_data.jsonl",
            "--api_key", "ollama",
            "--api_model", "qwen3-vl",
            "--max_iter", "15",
            "--max_attached_imgs", "1",
            "--temperature", "0.1",
            "--headless",
            "--output_dir", "results/Exp2_NoColor"
        ]
    },
    # Add your new invented experiments here...
]

for exp in experiments:
    print(f"\n\n==================================================")
    print(f"STARTING EXPERIMENT: {exp['name']}")
    print(f"==================================================\n")
    
    start_time = time.time()
    
    # Run the command
    try:
        # We use subprocess to run the command and wait for it to finish
        result = subprocess.run(exp['cmd'], check=False)
        
        if result.returncode == 0:
            print(f"Experiment {exp['name']} finished successfully.")
        else:
            print(f"Experiment {exp['name']} failed with code {result.returncode}.")
            
    except Exception as e:
        print(f"Critical error running experiment {exp['name']}: {e}")
        
    duration = time.time() - start_time
    print(f"Duration: {duration:.2f} seconds")
    
    # Optional: Cool down GPU
    time.sleep(5) 

print("All experiments completed.")