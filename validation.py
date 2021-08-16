import torch
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_recall_fscore_support


def validation(dataloader, model, loss_fn, device, use_cnn=False):
    size = len(dataloader)
    model.eval()
    val_loss, correct, sum = 0, 0, 0

    listy = []
    listp = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.long)
            if use_cnn:
                pred = model(X.reshape(-1, 4, 5))
            else:
                pred = model(X)

            loss = loss_fn(pred, y)
            val_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            sum += X.shape[0]
            listy += y.tolist()
            tmp = pred.argmax(1)
            listp += pred.argmax(1).tolist()

    CM = confusion_matrix(listy, listp)
    print(CM)
    val_loss /= size
    correct /= sum
    print(f"Validation Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {val_loss:>8f}")
    print("F1 Score: {} \n".format(f1_score(listy, listp, average='macro')))
    # class_names = ['经营风险', '法律风险', '贷款风险', '其他风险', '暂无风险']
    # class_names = ['经营风险', '法律风险', '贷款风险', '其他风险']
    print(classification_report(listy, listp))
    # precision, recall, f_score, support = precision_recall_fscore_support(listy, listp, average='macro')
    # print("precision: {} \n".format(precision))
    # print("recall: {} \n".format(recall))
    # print("fbeta_score: {} \n".format(f_score))
