# --- START OF FILE auto_labeler_final_v2.py ---

import os
import json
import csv
import base64
import time
import re
from collections import Counter
from openai import OpenAI

# --- CONFIGURATION ---
RESULTS_DIR = "results/Exp1_Baseline/20251207_22_56_39"
OUTPUT_CSV = "master_dataset_v6.csv" # The definitive dataset
MODEL_NAME = "qwen3-vl:30b"
API_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
MAX_RETRIES = 3
SLEEP_INTERVAL = 0.2
# --- CRITICAL ---
API_TIMEOUT = 120.0 # Wait up to 120 seconds for a response from the model

# --- FINAL TAXONOMY ---
LABELS = [
    "HOMEPAGE", "SEARCH_RESULTS_LISTING", "CATEGORY_LISTING", "DETAIL_PAGE",
    "FORM_PAGE", "MAP_VIEW", "ARTICLE_DOCUMENT", "POPUP_OVERLAY",
    "NAVIGATION_MENU_ACTIVE", "SEARCH_OVERLAY_ACTIVE", "FILTER_SORT_OPTIONS",
    "LOGIN_SIGNUP", "CAPTCHA_SECURITY", "ERROR_PAGE", "LOADING_STATE", "OTHER",
    "MODEL_ERROR", "IMAGE_READ_ERROR"
]

client = OpenAI(base_url=API_URL, api_key=API_KEY)

# --- A System Prompt makes the model more reliable ---
SYSTEM_PROMPT = f"""
You are a precise Visual Analyst Bot. Your task is to analyze a screenshot and classify its visual state by thinking step-by-step.

**CRITICAL INSTRUCTIONS:**
1.  Your entire response MUST be a single, valid JSON object and nothing else.
2.  Do not include any conversational text or markdown formatting like ```json before or after the JSON object.
3.  The JSON object must contain two keys: "chain_of_thought" and "label".
4.  The `chain_of_thought` value must be a brief, step-by-step analysis of the visual evidence.
5.  The `label` value must be EXACTLY ONE of the following allowed labels: {', '.join(LABELS)}

**EXAMPLE OUTPUT FORMAT:**
{{"chain_of_thought": "Step 1: The image shows a webpage with readable content in the background. Step 2: A large dialog box is superimposed on top, asking for 'Privacy' choices. Step 3: This matches the definition of a popup.", "label": "POPUP_OVERLAY"}}
"""

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"ERROR: Failed to read image {image_path}: {e}")
        return None

def classify_image_with_retry(image_path):
    base64_image = encode_image(image_path)
    if not base64_image:
        return "IMAGE_READ_ERROR", "Could not read the image file from disk."
    
    user_prompt = "Analyze the provided screenshot and provide your response in the specified JSON format."

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ],
                max_tokens=300,
                temperature=0.0,
                timeout=API_TIMEOUT # <-- APPLYING THE TIMEOUT
            )
            response_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                print(f"  WARNING: No JSON object found in response on attempt {attempt + 1}/{MAX_RETRIES}. Response: '{response_text[:150]}...'")
                time.sleep(1)
                continue

            data = json.loads(json_match.group(0))
            label = data.get("label")
            reasoning = data.get("chain_of_thought", "No reasoning provided.")
            
            if label in LABELS:
                return label, reasoning
            else:
                print(f"  WARNING: Model returned an invalid label '{label}' in JSON on attempt {attempt + 1}/{MAX_RETRIES}.")
                time.sleep(1)

        except json.JSONDecodeError:
            print(f"  WARNING: Failed to decode JSON on attempt {attempt + 1}/{MAX_RETRIES}. Response: '{response_text[:150]}...'")
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR: API call failed on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(2)

    return "MODEL_ERROR", f"Failed to get a valid classification after {MAX_RETRIES} attempts."

def main():
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label", "chain_of_thought", "task_id", "task_status"])

    processed_paths = get_processed_images(OUTPUT_CSV)
    print(f"Found {len(processed_paths)} already labeled images. Scanning for new ones in '{RESULTS_DIR}'...")
    newly_labeled_stats = Counter()

    all_png_files = []
    for root, _, files in os.walk(RESULTS_DIR):
        for file in files:
            if file.endswith(".png"):
                all_png_files.append((root, file))

    total_files = len(all_png_files)
    for i, (root, file) in enumerate(all_png_files):
        full_path = os.path.normpath(os.path.join(root, file))
        if full_path in processed_paths:
            continue
            
        summary_path = os.path.join(root, "summary.json")
        task_status = "Running/Unknown"
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    task_status = json.load(f).get('status', 'Unknown')
            except Exception: pass
            
        task_id = os.path.basename(root)

        print(f"Labeling [{i+1}/{total_files}]: {task_id} / {file}...")
        label, reasoning = classify_image_with_retry(full_path)
        print(f" -> Label: {label}\n -> CoT: {reasoning}")
        newly_labeled_stats[label] += 1

        try:
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([full_path, label, reasoning, task_id, task_status])
            processed_paths.add(full_path)
        except Exception as e:
            print(f"CRITICAL: Error writing to CSV: {e}")
        
        time.sleep(SLEEP_INTERVAL)

    print("\n--- Final Run Summary ---")
    if not newly_labeled_stats:
        print("No new images were labeled in this run.")
    else:
        for label, count in newly_labeled_stats.most_common():
            print(f"  - {label}: {count} new labels")

def get_processed_images(csv_file):
    if not os.path.exists(csv_file): return set()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader, None)
            return set(os.path.normpath(row) for row in reader)
        except (StopIteration, IndexError): return set()

if __name__ == "__main__":
    main()
    print("\n\nLabeling process complete.")