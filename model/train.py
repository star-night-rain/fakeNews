from model.dataloader import load_training_data,load_test_data
from model.layer import *
# from dataloader import load_data
# from layer import *
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel
import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

model_path = "/home/team3/fakeNews/model/best_model.pth"


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

        # evaluate the model on validation data
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
        print(
            f"Epoch {epoch+1} Done | Train Loss: {total_loss/len(train_dataloader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # save the best model based on validation loss
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

    # save the best model after training
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


import random
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=100,
    lr=1e-5,
    device="cuda",
   save_path="/content/drive/MyDrive/colab/best_model.pth",
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

            loss = loss_main +  1.5*loss_bg + 1.5* loss_iss
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
            torch.save(best_model_state_dict, save_path)
            print(f"Validation loss improved. Saving model to {save_path}")
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

    # Save the best model after training
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, save_path)
        print(f"Best model saved to {save_path}")


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
                loss_main.item() + 1.5*loss_bg.item() + 1.5*loss_iss.item()
            )

            pred = torch.sigmoid(output) >= 0.5
            correct += (pred == labels.bool()).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc


# def evaluate_model(model, dataloader, criterion, device="cuda"):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             bg_pred, iss_pred, output = model(batch)
#             labels = batch["label"].float().view(-1, 1)

#             loss_bg = criterion(bg_pred, labels)
#             loss_iss = criterion(iss_pred, labels)
#             loss_main = criterion(output, labels)

#             total_loss += (
#                 loss_main.item() + 1.5 * loss_bg.item() + 1.5 * loss_iss.item()
#             )

#             preds = (torch.sigmoid(output) >= 0.5).int()
#             # print(preds)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy().astype(int))

#             correct += (preds == labels.bool()).sum().item()
#             total += labels.size(0)

#     avg_loss = total_loss / len(dataloader)
#     acc = correct / total

#     precision = precision_score(all_labels, all_preds, zero_division=0)
#     recall = recall_score(all_labels, all_preds, zero_division=0)
#     f1 = f1_score(all_labels, all_preds, zero_division=0)

#     # 每个类别的 precision, recall, f1, support
#     per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
#         all_labels, all_preds, zero_division=0
#     )

#     # 每个类别的 accuracy
#     labels_np = np.array(all_labels)
#     preds_np = np.array(all_preds)

#     pos_idx = labels_np == 1
#     neg_idx = labels_np == 0

#     pos_acc = (preds_np[pos_idx] == 1).sum() / pos_idx.sum() if pos_idx.sum() > 0 else 0
#     neg_acc = (preds_np[neg_idx] == 0).sum() / neg_idx.sum() if neg_idx.sum() > 0 else 0

#     return {
#         "avg_loss": avg_loss,
#         "overall": {
#             "accuracy": acc,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         },
#         "per_class": {
#             "negative": {
#                 "accuracy": neg_acc,
#                 "precision": per_class_prec[0],
#                 "recall": per_class_rec[0],
#                 "f1": per_class_f1[0]
#             },
#             "positive": {
#                 "accuracy": pos_acc,
#                 "precision": per_class_prec[1],
#                 "recall": per_class_rec[1],
#                 "f1": per_class_f1[1]
#             }
#         }
#     }


def inference(news):
    bert_model = BertModel.from_pretrained("/home/team3/huggingface/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea", local_files_only=True)
    model = FakeNewDetecter(bert_model)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        _, __, output = model(news)
    return output.item()

if __name__ == "__main__":
    pretrained_model_file = '/home/team3/huggingface/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f'
    bert_model = BertModel.from_pretrained(pretrained_model_file, local_files_only=True)
    model = FakeNewDetecter(bert_model)
    train_dataloader, val_dataloader, test_dataloader = load_training_data()

    train_model(model, train_dataloader, val_dataloader)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    device = 'cuda'

    criterion = nn.BCEWithLogitsLoss()

    loss, acc = evaluate_model(model, test_dataloader, criterion, device)
    print(f'average_loss:{loss:.2f},accuracy:{acc:.2f}')
