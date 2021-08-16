from model import NeuralNetwork
from data import read_test_data

import torch
from torch.nn.functional import softmax
import argparse
import json
import os
from collections import defaultdict


def predict(x, model, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x).to(device).to(torch.float32)
        pred = softmax(model(x), dim=1)

    return pred.argmax(1), pred


def save_res(pred, cor_dict):
    res = []
    for k, v in list(cor_dict.items())[:-1]:
        #     print(k)
        tem = [0] * 5
        for idx in v:
            tem[pred[idx]] += 1
        # print(tem.index(max(tem)))
        res.append((k, tem.index(max(tem))))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str,
                        default="./model/final_cleaned1_trained_after_data_argument28_model6555.pth")
    parser.add_argument('-test_data', type=str, default="./data/testing_data/Constructed-28/final_constructed1.csv")
    parser.add_argument('-save_dir', type=str, default="./data/testing_data/Constructed-28/")
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-model_num', type=int, default=5)
    parser.add_argument('-input_size', type=int, default=22)
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-output_size', type=int, default=5)
    parser.add_argument('-submodel_output_size', type=int, default=16)
    parser.add_argument('-dropout_p', type=float, default=0.2)
    parser.add_argument('-batch_size', type=int, default=32)

    args = parser.parse_args()

    save_path_pred = args.save_dir + 'pred_' + os.path.basename(args.test_data)
    save_path_pvec = args.save_dir + 'pvec_' + os.path.basename(args.test_data)
    save_path_fres = args.save_dir + 'fres2_' + os.path.basename(args.test_data)[:-4] + '.txt'
    args.device = args.device if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(args.device))

    model = NeuralNetwork(device=args.device,
                          model_num=args.model_num,
                          input_size=args.input_size,
                          hidden_size=args.hidden_size,
                          output_size=args.output_size,
                          submodel_output_size=args.submodel_output_size,
                          dropout_p=args.dropout_p,
                          batch_size=args.batch_size
                          ).to(args.device)

    model.load_state_dict(torch.load(args.model_path))

    data, ids, cor_dict = read_test_data(args.test_data)
    res, pvec = predict(data, model, device=args.device)

    fres = save_res(res, cor_dict)
    # cnt = defaultdict(int)
    # for item in res.tolist():
    #     cnt[item] += 1
    # print(cnt)

    with open(save_path_pred, 'w') as fd1, open(save_path_pvec, 'w') as fd2, open(save_path_fres, 'w') as fd3:
        fd1.write(json.dumps(res.tolist()))
        fd1.close()
        fd2.write(json.dumps(pvec.tolist()))
        fd2.close()
        fd3.write(json.dumps(fres))
        fd3.close()
    print("Done!")
