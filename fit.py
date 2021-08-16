from model import NeuralNetwork
from train import train
from data import data_split, get_dataset
from validation import validation
from CNN_Model import CNNNeuralNetwork

import argparse
import torch
from torch import nn
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', type=bool, default=False)
    parser.add_argument('-data_path', type=str, default="")
    parser.add_argument('-train_data_path', type=str, default="./data/training_data/train.csv")
    parser.add_argument('-val_data_path', type=str, default="./data/validation_data/val.csv")

    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-val_p', type=float, default=0.2)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-model_num', type=int, default=5)
    parser.add_argument('-input_size', type=int, default=10)
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-output_size', type=int, default=5)
    parser.add_argument('-submodel_output_size', type=int, default=16)
    parser.add_argument('-dropout_p', type=float, default=0.2)
    parser.add_argument('-random_seed', type=int, default=9000)
    parser.add_argument('-model_save_dir', type=str, default='./model/')
    parser.add_argument('-train_balance', type=bool, default=True)
    parser.add_argument('-atten_flag', type=bool, default=True)
    parser.add_argument('-use_cnn', type=bool, default=False)

    args = parser.parse_args()
    model_save_path = args.model_save_dir + os.path.basename(args.data_path)[:-4] + \
                      "_trained_after_data_argument28_model" + str(args.random_seed) + ".pth"

    # Get cpu or gpu device for training.
    args.device = args.device if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(args.device))

    if not args.use_cnn:
        model = NeuralNetwork(device=args.device,
                              model_num=args.model_num,
                              input_size=args.input_size,
                              hidden_size=args.hidden_size,
                              output_size=args.output_size,
                              submodel_output_size=args.submodel_output_size,
                              dropout_p=args.dropout_p,
                              batch_size=args.batch_size,
                              atten_flag=args.atten_flag
                              ).to(args.device)
    else:
        model = CNNNeuralNetwork(model_num=args.model_num,
                                 input_size=args.input_size,
                                 in_channel=1,
                                 out_channel=20,
                                 block_out_channel=10,
                                 kernel_size=3,
                                 stride=1,
                                 num_layers=2,
                                 dropout_p=0.1
                                 ).to(args.device)
    # print(model)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.data_type:
        train_dataloader, val_dataloader = data_split(args.data_path, args.batch_size,
                                                      val_p=args.val_p, random_seed=args.random_seed,
                                                      train_balance=args.train_balance)
    else:
        train_dataloader, val_dataloader = get_dataset(args.train_data_path, args.val_data_path, args.batch_size)

    print('Training Data Size: ', len(train_dataloader) * args.batch_size)
    print('Validation Data Size: ', len(val_dataloader) * args.batch_size)

    for t in range(args.epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(dataloader=train_dataloader, model=model, loss_fn=loss_fn,
              optimizer=optimizer, device=args.device, use_cnn=args.use_cnn)
        validation(dataloader=val_dataloader, model=model, loss_fn=loss_fn, device=args.device, use_cnn=args.use_cnn)
    torch.save(model.state_dict(), model_save_path)
    print("Saved PyTorch Model State to model.pth: {}".format(model_save_path))
    print("Done!")
