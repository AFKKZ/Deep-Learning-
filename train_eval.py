import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from model import TransformerModel
from data import load_data
import time

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
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return {'accuracy': acc, 'f1_score': f1}

def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       epochs=100, lr=0.0005, eval_epochs=[10, 30, 50, 100],
                       model_name="transformer", early_stop_patience=3):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    results = {"validation": {}, "test": {}, "timing": {}}
    config_name = "transformer"
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

    with open(f"{model_name}_{config_name}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"{model_name} results saved to {model_name}_{config_name}_results.json")
    return results

def main():
    embed_dim = 100
    hidden_dim = 128
    output_dim = 3
    dropout = 0.2
    batch_size = 32
    epochs = 100
    eval_epochs = [10, 30, 50, 100]

    train_loader, val_loader, test_loader, vocab, label_map, pretrained_embeddings = load_data()
    vocab_size = len(vocab)
    pad_idx = vocab['<PAD>']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        nhead=4,
        num_layers=2,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        pad_idx=pad_idx,
        context_vocab_size=len(vocab),
        pretrained_embeddings=pretrained_embeddings
    )

    train_and_evaluate(
        model, train_loader, val_loader, test_loader, device,
        epochs=epochs,
        eval_epochs=eval_epochs,
        model_name="transformer",
        early_stop_patience=3
    )

if __name__ == "__main__":
    main()
