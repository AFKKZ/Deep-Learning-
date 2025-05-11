import os
import json

# Define your experiment folders and associated notes
experiment_configs = {
    "PreT_NoAtt_GloVe": {
        "label": "Pretrained + No Attn",
        "notes": "Uses GloVe embeddings only"
    },
    "PreT_Att_WeightedComb": {
        "label": "Pretrained + Attn",
        "notes": "GloVe + attention mechanism"
    },
    "Scratch_NoAtt_random_embeddings": {
        "label": "Scratch + No Attn",
        "notes": "Random embeddings trained from scratch"
    },
    "Scratch_Att_CustomAtt": {
        "label": "Scratch + Attn",
        "notes": "Random embeddings with attention"
    }
}

# Where all the result folders live
base_dir = "Results"

# Prepare the final output table data
table_data = []

for folder, meta in experiment_configs.items():
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping missing folder: {folder_path}")
        continue

    # Try to find the JSON file in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            json_path = os.path.join(folder_path, file)
            with open(json_path, 'r') as f:
                results = json.load(f)

            test_metrics = results.get("test", {}).get("epoch_100", {})
            accuracy = test_metrics.get("accuracy", 0)
            f1 = test_metrics.get("f1_score", 0)

            table_data.append([
                meta["label"],
                f"{accuracy:.4f}",
                f"{f1:.4f}",
                meta["notes"]
            ])
            break

# Print the results in table form
print("\n+---------------------------+------------+-----------+-----------------------------------------+")
print("| Experiment                | Accuracy   | F1-score  | Notes                                   |")
print("+---------------------------+------------+-----------+-----------------------------------------+")
for row in table_data:
    print(f"| {row[0]:<25} | {row[1]:<10} | {row[2]:<9} | {row[3]:<29} |")
print("+---------------------------+------------+-----------+-----------------------------------------+")
