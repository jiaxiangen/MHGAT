import networkx as nx
import numpy as np
import scipy
import pickle
import dgl
import torch
def load_DOUBAN_929_data(prefix='data/douban',train_val_test_dir = '/'):

    features_0 = np.load(prefix + '/features_douban_928_1_movie.npz.npy')
    features_1 =np.load(prefix + '/features_douban_928_1_directors.npz.npy')
    features_2= np.load(prefix + '/features_douban_928_1_actor.npz.npy')
    node_test = features_0.shape[0]

    #img
    features_0_img = np.load(prefix + '/img_feature_douban_512_199.npz')['feature']
    # features_0 = np.load(prefix + '/img_feature_20_3_25_199.npz')['feature']
    # features_0 = np.load(prefix + '/features_douban_908_2_actor_0.npz.npy')
    features_1_img = np.load(prefix + '/features_douban_928_1_directors.npz.npy')
    features_2_img =np.load(prefix + '/features_douban_928_1_actor.npz.npy')

    labels = np.load(prefix + '/labels.npy')#labels
    train_val_test_idx = np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/douban_actor_rdf.npy')
    return [features_0, features_1, features_2],[features_0_img, features_1_img, features_2_img],node_test,labels,train_val_test_idx,rdf

def load_AMAZON_data(prefix='data/amazon',train_val_test_dir = '/'):

    features_0 = np.load(prefix + '/feature_item_amazon.npz')['feature']
    features_1 =np.load(prefix + '/feature_review_amazon.npz')['feature']
    node_test = features_0.shape[0]
    #img
    features_0_img = np.load(prefix + '/img_feature_20_1_25_199.npz')['feature']
    features_1_img = np.load(prefix + '/feature_review_amazon.npz')['feature']

    labels = np.load(prefix + '/labels.npy')#labels
    train_val_test_idx = np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/rdf.npy')
    return [features_0, features_1],[features_0_img, features_1_img],node_test,labels,train_val_test_idx,rdf

def load_IMDB_data(prefix='data/imdb',train_val_test_dir = '/'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').todense()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').todense()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').todense()
    node_test = features_0.shape[0]
    #img
    features_0_img = np.load(prefix + '/img_feature_20_6_25_199.npz')['feature']
    features_1_img = scipy.sparse.load_npz(prefix + '/features_1.npz').todense()
    features_2_img = scipy.sparse.load_npz(prefix + '/features_2.npz').todense()

    type_mask = np.load(prefix + '/node_types.npy')#node labels
    labels = np.load(prefix + '/labels.npy')#labels
    train_val_test_idx = np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/rdf.npy')
    return [features_0, features_1, features_2],[features_0_img, features_1_img, features_2_img],node_test ,labels,train_val_test_idx,rdf
