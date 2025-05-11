import os
import json
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.join(os.path.dirname(__file__), "Results")

configs = {
    "PreT_NoAtt_GloVe": "Pretrained + No Attn (GloVe)",
    "PreT_Att_WeightedComb": "Pretrained + Attn (Weighted)",
    "Scratch_NoAtt_random_embeddings": "Scratch + No Attn",
    "Scratch_Att_CustomAtt": "Scratch + Attn (Custom)"
}

def load_model_scores(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    acc = data['test']['epoch_100']['accuracy']
    f1 = data['test']['epoch_100']['f1_score']
    time = data['timing']['epoch_100']
    return acc, f1, time

def gather_all_results():
    labels = []
    cnn_acc, lstm_acc = [], []
    cnn_f1, lstm_f1 = [], []
    cnn_time, lstm_time = [], []

    for folder, label in configs.items():
        path = os.path.join(base_dir, folder)
        labels.append(label)

        # CNN+LSTM results
        cnn_json = next((f for f in os.listdir(path) if f.startswith("cnn_lstm") and f.endswith(".json")), None)
        cnn_path = os.path.join(path, cnn_json)
        acc, f1, time = load_model_scores(cnn_path)
        cnn_acc.append(acc)
        cnn_f1.append(f1)
        cnn_time.append(time)

        # LSTM-only results
        lstm_json = next((f for f in os.listdir(path) if f.startswith("lstm_only") and f.endswith(".json")), None)
        lstm_path = os.path.join(path, lstm_json)
        acc, f1, time = load_model_scores(lstm_path)
        lstm_acc.append(acc)
        lstm_f1.append(f1)
        lstm_time.append(time)

    return labels, cnn_acc, lstm_acc, cnn_f1, lstm_f1, cnn_time, lstm_time

def plot_grouped_bar(cnn_values, lstm_values, labels, ylabel, title, filename):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, lstm_values, width, label="LSTM Only", color="#4C72B0")
    bars2 = ax.bar(x + width/2, cnn_values, width, label="CNN + LSTM", color="#C44E52")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename))
    plt.show()

def plot_runtime_comparison(labels, cnn_time, lstm_time):
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, lstm_time, marker='o', label='LSTM Only', color='#55A868')
    ax.plot(x, cnn_time, marker='s', label='CNN + LSTM', color='#8172B3')

    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime Comparison Across Configurations", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "runtime_comparison.png"))
    plt.show()

if __name__ == "__main__":
    labels, cnn_acc, lstm_acc, cnn_f1, lstm_f1, cnn_time, lstm_time = gather_all_results()
    plot_grouped_bar(cnn_acc, lstm_acc, labels, "Test Accuracy", "Accuracy Comparison Across Settings", "accuracy_comparison.png")
    plot_grouped_bar(cnn_f1, lstm_f1, labels, "Test F1 Score", "F1 Score Comparison Across Settings", "f1_comparison.png")
    plot_runtime_comparison(labels, cnn_time, lstm_time)
