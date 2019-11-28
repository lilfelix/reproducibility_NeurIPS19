import encoder as en
import torch
import data_utils as du
import classification

import numpy as np
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV  # only this needed
from sklearn.svm import SVC


def test_causal_block():
    series = torch.randn(100, 10, 50)  # input should be (N,C,L)
    causal1 = en.CausalCNNLayer(
        in_channels=10, out_channels=10, kernel_size=3, dilation=1)
    out = causal1(series)
    # print(causal1)
    summary(model=causal1,
            input_size=(10, 50))  # input_size should not include N (batch size)
    print(out.size())


def test_encoder():
    seq_length = 50
    # input should be of shape (N,C,L) - N: samples, C: channels, L: length
    series = torch.randn(10, 1, seq_length)
    encoder = en.Encoder(seq_length=seq_length, n_layers=5, hidden_out_channels=40,
                         kernel_size=3, last_out_channels=320, rep_length=160)
    out = encoder(series)
    print(encoder)
    # summary(model=encoder, input_size=(1, 50))
    print("Encoder output shape is:", out.shape)
    # print(out)


def test_data_read(do_print=False):
    dataset_path = '../data/UCR/Coffee/Coffee_TRAIN.tsv'
    train_ds = du.TimeSeriesDatasetUnivariate(dataset_path=dataset_path, K=5)
    train_dataloader = DataLoader(train_ds, batch_size=5,
                                  shuffle=True, num_workers=1)

    if do_print:
        for i_batch, sample_batched in enumerate(train_dataloader):
            print(i_batch, sample_batched['label'],
                  sample_batched['x_ref'].size(),
                  sample_batched['x_pos'].size(),
                  sample_batched['label'].size())
    return train_dataloader


def test_encoder_with_data():
    train_dl = test_data_read(do_print=True)
    # IMPORTANT STEP BEFORE CALLING ENCODER: applying the .float() function after the encoder and data batches
    # this should be given according to the reference, positive, or negative sample size
    seq_length = 286
    encoder = en.Encoder(seq_length=seq_length,
                         n_layers=5, hidden_out_channels=40,
                         kernel_size=3, last_out_channels=320,
                         rep_length=160).float()

    for i_batch, batch in enumerate(train_dl):
        x_ref = torch.unsqueeze(batch['x_ref'].float(), dim=1)
        print("x_ref shape is: ", x_ref.size())
        print(encoder(x_ref).size())


def test_max_pooling():
    N, seq_length = 3, 10
    mx = torch.nn.MaxPool1d(kernel_size=seq_length)

    data = torch.randn(size=(N, 1, seq_length))
    out = mx(data)

    # print(data)
    print("outpus shape is:", out.size())
    # print(out)


def test_linear_transformation():
    N = 3
    last_out_channels = 320
    rep_length = 160
    # after max_pooling we have an [last_out_channels x 1] vector (e.g [320 x 1])
    max_pool_out = torch.randn(size=(N, last_out_channels, 1))
    linear = torch.nn.Linear(
        in_features=last_out_channels, out_features=rep_length)

    # to use nn.Linear() we need to bring the dimension that is reduced (320 -> 160) to the last index
    # see for why using .contiguous(): https://stackoverflow.com/questions/48915810/pytorch-contiguous
    max_pool_out = max_pool_out.permute(0, 2, 1).contiguous()
    out = linear(max_pool_out).permute(0, 2, 1).contiguous()
    print(out.size())


def test_svm():
    classifier = classification.SVMClassifier()
    representations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 1, 2])
    test_rep = [[3, 3]]
    # it accepts both np arrays and simple Python lists
    classifier.train(representations, labels)
    prediction = classifier.classify(test_rep)
    print(prediction)


def test_cross_validation():
    iris = datasets.load_iris()
    print(iris.data.shape, iris.target.shape)
    # print(type(iris.data))  # 'np array'

    # just as an example to have access to train and test representations and labels
    train_rep, test_rep, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.5)
    print('train data shape:', train_rep.shape, y_train.shape)
    print('test data shape:', test_rep.shape, y_test.shape)

    classifier = classification.SVMClassifier(train_rep, y_train)
    test_acc = classifier.classify(test_rep, y_test)
    print(test_acc)


def test_svm_on_representations():
    dataset = 'Coffee'
    k = 5
    classification.perform_svm_on_representations(dataset, k)


def test_svm_all():
    classification.perform_svm_all(rep_folder='features_red_purple_29oct', save_csv=True, recompute_all=False)


def main():
    # test_causal_block()
    # test_encoder()
    # test_data_read()
    # test_encoder_with_data()
    # test_max_pooling()
    # test_linear_transformation()
    # test_svm()
    # test_cross_validation()
    # test_svm_on_representations()
    print('Did you remember to change svm_results_1D name in with open(..)? and features path?')
    input()
    test_svm_all()


if __name__ == '__main__':
    main()

"""
Not a good place for putting this, but useful commands:
assumption: we are in the ~/PycharmProjects/reproducibility_NIPS19 directory

INITIAL RSYNC from  ~/PycharmProjects/ (no longer to be used):
rsync -Pav -e "ssh -i ~/.ssh/gcp-key" reproducibility_NIPS19/ m.moein.sorkhei@35.209.222.133:/home/m.moein.sorkhei/reproducibility_NIPS19/



SSH: 
ssh -i ~/.ssh/gcp-key m.moein.sorkhei@35.209.222.133
ssh -i ~/.ssh/gcp-key m.moein.sorkhei@35.233.149.209


RSYNC CODE FROM HERE:
rsync -Pav -e "ssh -i ~/.ssh/gcp-key" src/ m.moein.sorkhei@35.209.222.133:/home/m.moein.sorkhei/reproducibility_NIPS19/src/
rsync -Pav -e "ssh -i ~/.ssh/gcp-key" src/ m.moein.sorkhei@35.233.149.209:/home/m.moein.sorkhei/reproducibility_NIPS19/src/



RSYNC DATA FROM HERE:
rsync -Pav -e "ssh -i ~/.ssh/gcp-key" data/ m.moein.sorkhei@35.209.222.133:/home/m.moein.sorkhei/reproducibility_NIPS19/data/
rsync -Pav -e "ssh -i ~/.ssh/gcp-key" data/household_power_consumption/ m.moein.sorkhei@35.233.149.209:/home/m.moein.sorkhei/reproducibility_NIPS19/data/household_power_consumption/


RSYNC MODEL FROM HERE:
rsync -Pav -e "ssh -i ~/.ssh/gcp-key" IHEPC_models/ m.moein.sorkhei@35.233.149.209:/home/m.moein.sorkhei/reproducibility_NIPS19/IHEPC_models/


RSYNC RESULTS FROM THERE (now only syncing the svm file):
rsync -Pav -e "ssh -i ~/.ssh/gcp-key" m.moein.sorkhei@35.233.149.209:/home/m.moein.sorkhei/reproducibility_NIPS19/IHEPC_models/day/ IHEPC_models/day/



SCREEN functioanlity:
- type screen
- start the process
- Ctrl+A and then Ctrl+D
- for resuming: screen -r
"""