import json
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from model import TransformerModel
from data import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, context, labels in data_loader:
            text, context, labels = text.to(device), context.to(device), labels.to(device)
            outputs = model(text, context)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')  # Using macro averaging as per requirements
    return {'accuracy': acc, 'f1_score': f1}

def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       epochs=100, lr=0.0005, eval_epochs=[10, 30, 50, 100],
                       output_dir="Results", model_name="transformer", early_stop_patience=3):

    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    results = {"validation": {}, "test": {}, "timing": {}}
    start_time = time.time()

    best_f1 = 0.0
    patience_counter = 0
    best_epoch = 0
    best_val_result, best_test_result, best_runtime = None, None, None

    for epoch in range(1, epochs + 1):
        model.train()
        for text, context, labels in train_loader:
            text, context, labels = text.to(device), context.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(text, context)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_result = evaluate_model(model, val_loader, device)
        val_f1 = val_result['f1_score']

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_val_result = val_result
            best_test_result = evaluate_model(model, test_loader, device)
            best_runtime = time.time() - start_time
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_best.pth"))
        else:
            patience_counter += 1
            print(f"[{model_name}] No improvement (patience {patience_counter}/{early_stop_patience})")

        if epoch in eval_epochs:
            elapsed = time.time() - start_time
            test_result = evaluate_model(model, test_loader, device)
            results["validation"][f"epoch_{epoch}"] = val_result
            results["test"][f"epoch_{epoch}"] = test_result
            results["timing"][f"epoch_{epoch}"] = elapsed
            print(f"[{model_name}] Epoch {epoch} - Val Acc: {val_result['accuracy']:.4f}, F1: {val_result['f1_score']:.4f} | "
                  f"Test Acc: {test_result['accuracy']:.4f}, F1: {test_result['f1_score']:.4f} - Time: {elapsed:.2f}s")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} (best F1 = {best_f1:.4f})")
            break

    if not results["validation"] and best_val_result is not None:
        results["validation"][f"epoch_{best_epoch}"] = best_val_result
        results["test"][f"epoch_{best_epoch}"] = best_test_result
        results["timing"][f"epoch_{best_epoch}"] = best_runtime
        print(f"[{model_name}] Saved best checkpoint at epoch {best_epoch}")

    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"{model_name} results saved to {output_file}")
    return results

def run_experiment(config_name, use_pretrained, use_cross_attention, batch_size=32):
    """
    Run a single experiment with the given configuration
    """
    # Create output directory
    output_dir = os.path.join("Results", config_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader, vocab, label_map, pretrained_embeddings = load_data(batch_size=batch_size)
    
    # Configure model parameters
    embed_dim = 100
    hidden_dim = 128
    output_dim = 3  # Positive, Negative, Neutral
    dropout = 0.2
    pad_idx = vocab['<PAD>']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = TransformerModel(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        nhead=4,
        num_layers=2,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        pad_idx=pad_idx,
        context_vocab_size=len(vocab),
        pretrained_embeddings=pretrained_embeddings,
        use_cross_attention=use_cross_attention,
        use_pretrained=use_pretrained
    )
    
    # Log configuration
    print(f"\n{'='*50}")
    print(f"Running experiment: {config_name}")
    print(f"Pretrained: {use_pretrained}, Cross-Attention: {use_cross_attention}")
    print(f"{'='*50}\n")
    
    # Train and evaluate
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_name=config_name,
        early_stop_patience=3
    )
    
    # Log completion
    print(f"\n{'='*50}")
    print(f"Completed experiment: {config_name}")
    print(f"{'='*50}\n")
    
    return output_dir

def generate_summary_table():
    # Define your experiment folders and associated notes
    experiment_configs = {
        "PreT_Concat_GloVe": {
            "label": "Pretrained + Concat",
            "notes": "Uses GloVe embeddings"
        },
        "PreT_CrossAtt": {
            "label": "Pretrained + Cross-Attn",
            "notes": "Context attends to text"
        },
        "Scratch_Concat": {
            "label": "Scratch + Concat",
            "notes": "Random embeddings"
        },
        "Scratch_CrossAtt": {
            "label": "Scratch + Cross-Attn",
            "notes": "Custom attention"
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

                # Try different epoch keys, starting with 100
                test_metrics = {}
                for epoch_key in ['epoch_100', 'epoch_50', 'epoch_30', 'epoch_10']:
                    if epoch_key in results.get("test", {}):
                        test_metrics = results.get("test", {}).get(epoch_key, {})
                        break

                # If no expected epoch keys are found, use the last available epoch
                if not test_metrics:
                    last_epoch_key = max(results.get("test", {}).keys(), default=None)
                    if last_epoch_key:
                        test_metrics = results.get("test", {}).get(last_epoch_key, {})

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

    # Export to CSV for easier inclusion in reports
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "experiment_results.csv")
    with open(csv_path, "w") as f:
        f.write("Experiment,Accuracy,F1-score,Notes\n")
        for row in table_data:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

    print(f"\nResults saved to CSV: {csv_path}")

def plot_results():
    """
    Load experiment results and generate plots
    """
    base_dir = "Results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Experiment configurations
    experiment_configs = {
        "PreT_Concat_GloVe": "Pretrained + Concat",
        "PreT_CrossAtt": "Pretrained + Cross-Attn",
        "Scratch_Concat": "Scratch + Concat",
        "Scratch_CrossAtt": "Scratch + Cross-Attn"
    }
    
    results_data = []
    
    for folder_name, display_name in experiment_configs.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Missing folder {folder_path}")
            continue
            
        # Find results JSON file
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        if not json_files:
            print(f"Warning: No JSON result files in {folder_path}")
            continue
            
        # Load the first JSON file found
        with open(os.path.join(folder_path, json_files[0]), 'r') as f:
            data = json.load(f)
            
        # Try different epoch keys, starting with 100
        test_metrics = {}
        runtime = 0
        for epoch_key in ['epoch_100', 'epoch_50', 'epoch_30', 'epoch_10']:
            if epoch_key in data.get("test", {}):
                test_metrics = data.get("test", {}).get(epoch_key, {})
                runtime = data.get("timing", {}).get(epoch_key, 0)
                break

        # If no expected epoch keys are found, use the last available epoch
        if not test_metrics:
            last_epoch_key = max(data.get("test", {}).keys(), default=None)
            if last_epoch_key:
                test_metrics = data.get("test", {}).get(last_epoch_key, {})
                runtime = data.get("timing", {}).get(last_epoch_key, 0)
                
        accuracy = test_metrics.get("accuracy", 0)
        f1_score = test_metrics.get("f1_score", 0)
        
        # Add to results
        results_data.append({
            "name": display_name,
            "folder": folder_name,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "runtime": runtime,
            "embedding": "Pretrained" if folder_name.startswith("PreT") else "Scratch",
            "attention": "Cross-Attn" if "CrossAtt" in folder_name else "Concat"
        })
    
    if not results_data:
        print("No experiment results found. Run experiments first.")
        return
    
    # Plot accuracy comparison
    plot_comparison_bar_chart(
        results_data,
        metric="accuracy",
        title="Accuracy Comparison Across Configurations",
        filename=os.path.join(base_dir, "accuracy_comparison.png")
    )
    
    # Plot F1 score comparison
    plot_comparison_bar_chart(
        results_data,
        metric="f1_score",
        title="F1 Score Comparison Across Configurations",
        ylabel="F1 Score",
        filename=os.path.join(base_dir, "f1_comparison.png")
    )
    
    # Plot runtime comparison
    plot_runtime_comparison(results_data, base_dir)
    
    # Plot grouped comparisons
    plot_grouped_comparison(results_data, base_dir)
    
    print("All plots generated successfully!")

def plot_comparison_bar_chart(results_data, metric="accuracy", title=None, ylabel=None, filename=None):
    """
    Plot a bar chart comparing experiments
    """
    # Extract data for plotting
    labels = [r["name"] for r in results_data]
    values = [r[metric] for r in results_data]
    colors = ["#3274A1" if r["embedding"] == "Pretrained" else "#E1812C" for r in results_data]
    patterns = ["//" if r["attention"] == "Cross-Attn" else "" for r in results_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1)
    
    # Add patterns for cross-attention
    for bar, pattern in zip(bars, patterns):
        if pattern:
            bar.set_hatch(pattern)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Set labels and title
    ax.set_ylabel(ylabel or metric.capitalize())
    ax.set_title(title or f"{metric.capitalize()} Comparison")
    ax.set_ylim(0, max(values) * 1.15)  # Add some space above bars
    
    # Add legend for embedding types and attention mechanisms
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3274A1", edgecolor="black", label="Pretrained Embeddings"),
        Patch(facecolor="#E1812C", edgecolor="black", label="Random Embeddings"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Cross-Attention"),
        Patch(facecolor="white", edgecolor="black", label="Concatenation")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        
    plt.close()

def plot_runtime_comparison(results_data, base_dir):
    """
    Plot a bar chart comparing runtime
    """
    # Extract data for plotting
    labels = [r["name"] for r in results_data]
    runtimes = [r["runtime"] for r in results_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, runtimes, color="#C44E52", edgecolor="black", linewidth=1)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}s', ha='center', va='bottom')
    
    # Set labels and title
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Runtime Comparison")
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # Save
    filename = os.path.join(base_dir, "runtime_comparison.png")
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    
    plt.close()

def plot_grouped_comparison(results_data, base_dir):
    """
    Plot grouped bar charts comparing embedding types and attention mechanisms
    """
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(results_data)
    
    # Plot metrics by embedding type (Pretrained vs Scratch)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by embedding type
    embedding_groups = df.groupby("embedding")
    
    # Plot accuracy by embedding type
    embedding_groups["accuracy"].mean().plot(kind="bar", ax=ax1, color=["#3274A1", "#E1812C"], 
                                   yerr=embedding_groups["accuracy"].std())
    ax1.set_title("Average Accuracy by Embedding Type")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    
    # Plot F1 by embedding type
    embedding_groups["f1_score"].mean().plot(kind="bar", ax=ax2, color=["#3274A1", "#E1812C"],
                                  yerr=embedding_groups["f1_score"].std())
    ax2.set_title("Average F1 Score by Embedding Type")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.0)
    
    # Add values on bars
    for ax, metric in zip([ax1, ax2], ["accuracy", "f1_score"]):
        for i, group in enumerate(embedding_groups.groups.keys()):
            height = embedding_groups[metric].mean()[group]
            ax.text(i, height + 0.02, f"{height:.3f}", ha="center")
    
    plt.tight_layout()
    filename = os.path.join(base_dir, "embedding_comparison.png")
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    
    plt.close()
    
    # Now compare attention mechanisms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by attention mechanism
    attn_groups = df.groupby("attention")
    
    # Plot accuracy by attention type
    attn_groups["accuracy"].mean().plot(kind="bar", ax=ax1, color=["#4C72B0", "#55A868"], 
                                   yerr=attn_groups["accuracy"].std())
    ax1.set_title("Average Accuracy by Attention Mechanism")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    
    # Plot F1 by attention type
    attn_groups["f1_score"].mean().plot(kind="bar", ax=ax2, color=["#4C72B0", "#55A868"],
                                  yerr=attn_groups["f1_score"].std())
    ax2.set_title("Average F1 Score by Attention Mechanism")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.0)
    
    # Add values on bars
    for ax, metric in zip([ax1, ax2], ["accuracy", "f1_score"]):
        for i, group in enumerate(attn_groups.groups.keys()):
            height = attn_groups[metric].mean()[group]
            ax.text(i, height + 0.02, f"{height:.3f}", ha="center")
    
    plt.tight_layout()
    filename = os.path.join(base_dir, "attention_comparison.png")
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis experiments")
    parser.add_argument('--step', type=str, choices=['train', 'evaluate', 'all'], 
                        default='all', help='Which step to run')
    parser.add_argument('--config', type=str, 
                        choices=['all', 'pretrained_concat', 'pretrained_crossatt', 'scratch_concat', 'scratch_crossatt'],
                        default='all', help='Which experiment configuration to run')
    args = parser.parse_args()
    
    # Create Results directory
    os.makedirs("Results", exist_ok=True)
    
    # Step 1: Train models
    if args.step == 'train' or args.step == 'all':
        print("\n=== STEP 1: Training Models ===")
        
        if args.config == "all" or args.config == "pretrained_concat":
            # Experiment 1: Pretrained Embeddings + Concatenation
            run_experiment(
                config_name="PreT_Concat_GloVe",
                use_pretrained=True,
                use_cross_attention=False
            )
        
        if args.config == "all" or args.config == "pretrained_crossatt":
            # Experiment 2: Pretrained Embeddings + Cross-Attention
            run_experiment(
                config_name="PreT_CrossAtt",
                use_pretrained=True,
                use_cross_attention=True
            )
        
        if args.config == "all" or args.config == "scratch_concat":
            # Experiment 3: From Scratch + Concatenation
            run_experiment(
                config_name="Scratch_Concat",
                use_pretrained=False,
                use_cross_attention=False
            )
        
        if args.config == "all" or args.config == "scratch_crossatt":
            # Experiment 4: From Scratch + Cross-Attention
            run_experiment(
                config_name="Scratch_CrossAtt",
                use_pretrained=False,
                use_cross_attention=True
            )
    
    # Step 2: Evaluate and generate reports
    if args.step == 'evaluate' or args.step == 'all':
        print("\n=== STEP 2: Evaluating and Generating Reports ===")
        generate_summary_table()
        plot_results()
        print("\nAll experiments completed!")
        print("Use the generated tables and plots for your final report.")

if __name__ == "__main__":
    main()