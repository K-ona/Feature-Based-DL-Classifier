import torch


def train(dataloader, model, loss_fn, optimizer, device, use_cnn=False):
    size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).to(torch.float32), y.to(device).to(torch.long)
        # print(X.shape[0])
        # predict
        if use_cnn:
            pred = model(X.reshape(-1, 4, 5))
        else:
            pred = model(X)
        # print(pred, y, sep='\t')
        # print(pred.shape, y.shape, sep='\t')
        loss = loss_fn(pred, y)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # print(mode)
        optimizer.step()

        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size * X.shape[0]:>5d}]")
