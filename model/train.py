from model.dataloader import load_data
from model.layer import *
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel

model_path = "./model/best_model.pth"


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=100,
    lr=1e-5,
    device="cuda",
):
    patience = 10
    cnt = 0
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")  # Initialize best validation loss as infinity
    best_model_state_dict = None  # To store the best model's state_dict

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            bg_pred, iss_pred, output = model(batch)
            labels = batch["label"].float()
            labels = labels.view(-1, 1)

            loss_bg = criterion(bg_pred, labels)
            loss_iss = criterion(iss_pred, labels)
            loss_main = criterion(output, labels)

            loss = loss_main + 1.5 * loss_bg + 1.5 * loss_iss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.sigmoid(output) >= 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        # Evaluate the model on validation data
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
        print(
            f"Epoch {epoch+1} Done | Train Loss: {total_loss/len(train_dataloader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, model_path)
            print(f"Validation loss improved. Saving model to {model_path}")
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

    # Save the best model after training
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, model_path)
        print(f"Best model saved to {model_path}")


def evaluate_model(model, dataloader, criterion, device="cuda"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            bg_pred, iss_pred, output = model(batch)
            labels = batch["label"].float()
            labels = labels.view(-1, 1)

            loss_bg = criterion(bg_pred, labels)
            loss_iss = criterion(iss_pred, labels)
            loss_main = criterion(output, labels)

            total_loss += (
                loss_main.item() + 1.5 * loss_bg.item() + 1.5 * loss_iss.item()
            )

            pred = torch.sigmoid(output) >= 0.5
            correct += (pred == labels.bool()).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc


def inference(news):
    bert_model = BertModel.from_pretrained("bert-base-chinese")
    model = FakeNewDetecter(bert_model)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        _, __, output = model(news)
    return output.item()


def main():
    bert_model = BertModel.from_pretrained("bert-base-chinese")
    model = FakeNewDetecter(bert_model)
    train_dataloader, val_dataloader, test_dataloader = load_data()

    train_model(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
