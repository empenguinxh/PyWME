#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import dask
import os
from dask.diagnostics import ProgressBar

import pyemd
import numpy as np
from scipy.spatial.distance import cdist
from numpy import linalg as LA

import gensim.corpora as corpora
import gensim.downloader as api

# number of processors to use for parallel processing
num_workers = 2
# seed = 2021
seed = np.random.randint(100000)

@dask.delayed
def get_wme(doc_wwv, doc_wev, doc_l_wwv_rand, doc_l_wev_rand, gamma, cache_fp=None):
    R = len(doc_l_wwv_rand)
    nuw = doc_wwv.shape[0]
    doc_wme = np.full(shape=R, fill_value=np.inf)
    for i in range(R):
        nuwr = doc_l_wwv_rand[i].shape[0]
        # get merged word weight vectors
        doc_wwv_ = np.concatenate([doc_wwv, np.zeros(nuwr)])
        doc_wwv_rand = np.concatenate([np.zeros(nuw), doc_l_wwv_rand[i]])
        # get distance matrix
        dist_mat = np.zeros((nuw+nuwr, nuw+nuwr))
        # only need to compute the cross document entries
        # of shape (nuw, nuwr)
        dist_mat_cd = cdist(doc_wev, doc_l_wev_rand[i], metric='euclidean')
        dist_mat[:nuw, nuw:(nuw+nuwr)] = dist_mat_cd
        dist_mat[nuw:(nuw+nuwr), :nuw] = dist_mat_cd.T
        # solve for wmd
        doc_wme_i = pyemd.emd(
            first_histogram=doc_wwv_, 
            second_histogram=doc_wwv_rand,
            distance_matrix=dist_mat)
        if gamma == -1:
            doc_wme[i] = doc_wme_i
        else:
            doc_wme[i] = np.exp(-doc_wme_i*gamma)
    doc_wme = doc_wme/np.sqrt(R)
    if cache_fp is not None:
        np.save(cache_fp, doc_wme)
    return doc_wme


def compute_wme(corpus_fp, R, D_min, D_max, gamma, nuw_max, wv_name, exp_id, no_cache):
    '''
    corpus_fp: path to the corpus file
    R: number of random documents
    D_min: minimum number of pseudo words that will be sampled in a random documents
    D_max: maximum number of pseudo words that will be sampled in a random documents
    gamma: 
       if gamma=-1, the original word mover's distance (WMD) will be used as wme features; 
       otherwise, exp(-gamma*WMD) will be used as wme features
    nuw_max: maximum number of most frequent unique words that will be kept in each document
    wv_name: 
      specify the pretrained word embedding matrix
      will be used as gensim.downloader.api.load(wv_name)
    no_cache: 
      whether to cache the computed WME vectors
    '''
    print(f"... loading word vectors: {wv_name} ...")
    w2v_data = api.load(wv_name)
    # derived hyperparameters
    val_max = w2v_data.vectors.max()
    val_min = w2v_data.vectors.min()
    wv_size = w2v_data.vector_size
    
    # ---------- prepare the corpus ----------
    corpus_name = os.path.splitext(os.path.split(corpus_fp)[1])[0]
    print(f"... loading corpus: {corpus_fp} ...")
    with open(corpus_fp, 'r') as f:
        corpus = [line.split() for line in f]
    dictionary = corpora.Dictionary(corpus)
    # `corpus` is a list, each element is the bag-of-word vector of a doc
    # and is stored as a list of tuples, each tuple is (word id, word count)
    # word id is mapped to word strings in `dictionary`
    corpus = [dictionary.doc2bow(doc) for doc in corpus]
    # sort unique words in each doc by their frequencies in the doc
    corpus = [sorted(doc, key=lambda x: x[1], reverse=True) for doc in corpus]
    n_uw_l = [len(doc) for doc in corpus]
    n_doc = len(corpus)
    print(f"... {n_doc} docs in total ...")
    
    rng = np.random.default_rng(seed)
    
    # ---------- generate random docs ----------
    print("... preparing random docs ...")
    # list of number of pseudo words in each doc, of shape [R,]
    n_uw_l_rand = rng.integers(D_min, D_max+1, size=R)
    # generate random word embedding vectors for all pseudo words
    doc_l_wev_rand = rng.uniform(low=val_min, high=val_max+1e-8, size=[n_uw_l_rand.sum(), wv_size])
    # to be consistent with pre-trained word vectors in the word2vec space
    # normalize each random word vector into an unit vector 
    doc_l_wev_rand = doc_l_wev_rand/LA.norm(doc_l_wev_rand, axis=1, keepdims=True)
    # distribute random word embedding vectors into random docs
    starts = np.concatenate([[0], n_uw_l_rand.cumsum()[:-1]])
    ends = starts+n_uw_l_rand
    doc_l_wev_rand = [doc_l_wev_rand[start:end] for start, end in zip(starts, ends)]
    # now, `doc_l_wev_rand[i]` is of shape `[n_uw_l_rand[i], wv_size]`
    # list of normalized word weight vectors for pseudo words in each doc
    doc_l_wwv_rand = [np.ones(n_uw)/n_uw for n_uw in n_uw_l_rand]
    
    # ---------- prepare docs ----------
    print("... preparing docs ...")
    doc_wwv_l = []
    doc_wev_l = []
    for doc_i in range(n_doc):
        doc_wwv = np.zeros(shape=n_uw_l[doc_i])
        doc_wev = np.zeros(shape=(n_uw_l[doc_i], wv_size))
        nuw = 0
        # allow words that is out-of-vocabulary in the word embedding matrix
        # to appear in a doc
        for i, (uw, uww) in enumerate(corpus[doc_i]):
            if i >= nuw_max:
                # to save computation time
                # less frequent words in a doc are not considered by WME
                break
            # `uw` means one unique word in the current doc, it is a token id
            # `uww` means the weight of `uw` in the current doc
            if dictionary[uw] in w2v_data:
                # `dictionary[uw]` gives the token string for `uw`
                # `w2v_data` can be regarded as a dictionary, 
                # with keys being token strings
                # and values being word embedding vectors
                doc_wwv[nuw] = uww
                doc_wev[nuw, :] = w2v_data[dictionary[uw]]
                nuw += 1
        doc_wwv = doc_wwv[:nuw]
        # normalize word weights
        doc_wwv = doc_wwv/doc_wwv.sum()
        doc_wev = doc_wev[:nuw]

        doc_wwv_l.append(doc_wwv)
        doc_wev_l.append(doc_wev)
    
    # ---------- compute WME vectors ----------
    print("... computing WME vectors...")
    wme_mat = np.zeros(shape=(n_doc, R))
    if no_cache:
        cache_fp_l = [None for doc_i in range(n_doc)]
    else:
        if not os.path.isdir('./tmp'):
            os.mkdir("./tmp")
        if not os.path.isdir(f"./tmp/{exp_id}"):
            os.mkdir(f"./tmp/{exp_id}")
        cache_fp_l = [f"./tmp/{exp_id}/{corpus_name}_{doc_i}_wme" for doc_i in range(n_doc)]
    skipped_l = []
    kept_l = []
    task_l = []
    for doc_i in range(n_doc):
        cache_fp = cache_fp_l[doc_i]
        if (not no_cache) and os.path.isfile(cache_fp+'.npy'):
            skipped_l.append(doc_i)
            wme_mat[doc_i] = np.load(cache_fp+'.npy')
        else:
            kept_l.append(doc_i)
            task_l.append(
                get_wme(
                    doc_wwv_l[doc_i], doc_wev_l[doc_i], 
                    doc_l_wwv_rand, doc_l_wev_rand, 
                    gamma, cache_fp)
            )
    if len(skipped_l) == n_doc:
        print(f"... load cache for all docs ...")
    elif len(skipped_l) > 0:
        print(f"... load cache for docs {skipped_l} ....")

    with ProgressBar():
        task_l = dask.compute(task_l, num_workers=num_workers, scheduler='processes')[0]
    for i, wme_vec in enumerate(task_l):
        wme_mat[kept_l[i]] = wme_vec
    # ---------- save ----------
    save_fp = f"./data/{corpus_name}_wme_{exp_id}"
    np.save(save_fp, wme_mat)
    print(f"saved as {save_fp}.npy")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute Word Mover's Embedding (WME) document vectors"
    )
    parser.add_argument('corpus_fp', 
                        metavar='str', type=str,
                        help='path to the corpus file, e.g., ./data/twitter_docs.txt')
    parser.add_argument('--gamma', 
                        metavar='float', type=float, default=1.0,
                        help="the \gamma parameter in paper; if gamma=-1, the original word mover's distance (wmd) will be used as wme features; otherwise, exp(-gamma*wmd) will be used")
    parser.add_argument('--R', 
                        metavar='N', type=int, default=128,
                        help='number of random documents, also the embedding dimension of WME')
    parser.add_argument('--D_max', 
                        metavar='N', type=int, default=6,
                        help='maximum number of pseudo words that will be sampled in a random documents')
    parser.add_argument('--D_min', 
                        metavar='N', type=int, default=1,
                        help='minimum number of pseudo words that will be sampled in a random documents')
    parser.add_argument('--nuw_max', 
                        metavar='N', type=int, default=500,
                        help='maximum number of most frequent unique words that will be kept in each document')
    parser.add_argument('--wv_name', 
                        metavar='str', type=str, default='word2vec-google-news-300',
                        help='specify the pretrained word embedding matrix. Check https://github.com/RaRe-Technologies/gensim-data for available options')
    parser.add_argument('--exp_id', 
                        metavar='str', type=str, default='exp_0',
                        help='experiment id')
    parser.add_argument('--no_cache', action='store_const', const=True, default=False,
                        help='whether to cache the computed WME vectors')
    
    args = parser.parse_args()

    print(f"experiment configuration: {args}")
    compute_wme(**vars(args))






