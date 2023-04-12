import process
import dgl
import networkx as nx
import torch
from model import Model

import argparse
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,roc_auc_score
import numpy as np
import numpy.random as random
import os
from utils import EarlyStopping

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed =0
def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(prediction, labels)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    return loss, accuracy, micro_f1, macro_f1


def load_g():
    features_list,labels, train_val_test_idx, rdf = process.load_AMAZON_data(
        train_val_test_dir='/train_val_test_idx_amaze.npz')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    features_list_img = process.load_AMAZON_Img_data()
    print(features_list_img[0].shape)
    print(features_list[1].shape)
    features_list_img = [torch.FloatTensor(features).to(device) for features in features_list_img]
    features_all = torch.cat((features_list), dim=0)
    features_img_all = torch.cat((features_list_img), dim=0)
    g = dgl.DGLGraph()
    # g.edata.update({'rel_type': edge_type, 'norm': edge_norm})
    edge_type = []
    g.add_nodes(13189)
    for x in rdf:
        # print(x)
        # print(x[0], x[1])
        g.add_edge(x[0], x[1])
        # t=0
        # if x[2]==1:
        #     x[2]=0
        #     t=0
        # elif x[2]==2:
        #     x[2]=1
        #     t=1
        # el
        if x[2]==3:
            x[2]=2
            t=2
        print(x[2])
        edge_type.append(x[2])
    print("*******")
    edge_type = torch.tensor(edge_type, dtype=torch.int64)
    # 求点的度
    _, edge_dsts = g.edges()
    print(g.in_degrees(edge_dsts))

    g.edata.update({'rel_type': edge_type})
    g.ndata['f'] = features_all
    g.ndata['f_img'] = features_img_all
    print(g)
    g = g.to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']

    val_idx = train_val_test_idx['val_idx']

    test_idx = train_val_test_idx['test_idx']
    # test_idx.append(val_idx)
    test_idx = list(test_idx)

    test_idx = test_idx + list(val_idx)
    test_idx = np.array(test_idx)
    print(test_idx)
    return g,train_idx,val_idx,test_idx,labels
def main(args):
    set_seed(seed)
    g, train_idx, val_idx, test_idx,labels= load_g()
    n_epochs = 100 # epochs to train
    lr = 0.001 # learning rate
    l2norm = 0 # L2 norm coefficient
    num_classes =3

    n_hidden_layers =3
    num_rels = 3
    maxMa = -1
    # create model
    model = Model(len(g),1433,
         64                                                      ,
                  num_classes,
                  num_rels,
                  num_hidden_layers=n_hidden_layers,dropout=0.6,mmGATdropout=0.6)
    # # optimizer

    model = model.to(device)
    print(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
    stopper = EarlyStopping(patience=30)
    loss_fcn = torch.nn.CrossEntropyLoss()
    print("start training...")

    for epoch in range(n_epochs):

        model.train()
        torch.cuda.empty_cache()
        # print(g)
        optimizer.zero_grad()
        logits= model.forward(g)

        loss = loss_fcn(logits[train_idx], labels[train_idx])


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_idx], labels[train_idx])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, labels, val_idx, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)
        test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, labels, test_idx, loss_fcn)
        print("Epoch {:05d} | ".format(epoch) +
              "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
                  train_acc, loss.item()) +
              "Validation Accuracy: {:.4f} | Validation loss: {:.4f}|".format(
                  val_acc, val_loss.item())+"test Accuracy: {:.4f} | test loss: {:.4f}".format(
                  test_acc, test_loss.item()))
        macro = test_macro_f1
        micro = test_micro_f1
        print("|test macro: {:.4f} |test micro: {:.4f} |".format(macro, micro))
        maxMa = max(maxMa, macro)
        if epoch % 10 == 0:
            print("maxcro{}".format(maxMa))
        print("maxcro{}".format(maxMa))
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, labels, test_idx, loss_fcn)
    print( "test Accuracy: {:.4f} | test loss: {:.4f}|test macro: {:.4f} |test micro: {:.4f} ".format(
        test_acc, test_loss.item(),test_macro_f1,test_micro_f1))


    model.embedding_out(g)

        # print(train_idx)
if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='MMRGAT')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    main(args)
