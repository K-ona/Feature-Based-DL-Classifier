### model description
model name: Self-Attention Based Multi-model Fusion Prediction Model.

self-attention mechanism is leveraged to fuse features got by sub-model.

model framework:
<img src=".\模型架构.png" alt= '示意图' style="zoom: 50%;" />

### train the model command: 
    python fit.py [-parameters]
parameters description: 

    -data_path 
        path of data used for fitting the model, e.g. ./data/trainingdata.csv
    -random_seed
        random seed
    -lr
        learning rate, default=0.01
    -val_p
        proportion of partition validation set, default=0.2
    -batch_size
        how many data batched for training, default=256:
    -device
        where to train the model, default='cuda'
    -model_num
        number of submodels(MLP\CNN etc. Classifier), default=5
    -input_size
        dimention of input featrue (number of column of input_file), default=28
    -hidden_size
        hidden state's dimention of linear layer, default=64
    -output_size
        whole model's output vector dimention, default=5
    -dropout_p
        probability applied by dropout layer, default=0.2
    -model_save_dir
        model parameter save dir, default='./model/'
    -train_balance
        whether to enhance data, default=True
### use model to predict
    python predict.py [-parameters]
parameters description:

     -model_path
        model path
     -test_data
        test data path
     -save_dir
        dir of prediction results, default="./data/testing_data/
     -input_size
        dimention of input featrue (number of column of input_file), default=28
        
the prediction result will be stored in save_dir
