from model.train import *

model_path = "/home/team3/fakeNews/model/best_model.pth"


if __name__ == "__main__":
    pretrained_model_file = '/home/team3/huggingface/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f'
    bert_model = BertModel.from_pretrained(pretrained_model_file, local_files_only=True)
    model = FakeNewDetecter(bert_model)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    device = 'cuda'

    criterion = nn.BCEWithLogitsLoss()

    dataloader = load_test_data(pretrained_model_file)

    loss, acc = evaluate_model(model, dataloader, criterion, device)
    print(f'average_loss:{loss:.2f},accuracy:{acc:.2f}')
