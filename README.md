# PyWME
A pure python implementation of the Word Mover‘s Embedding (WME) Algorithm (Wu et al. 2018). The [original implementation](https://github.com/IBM/WordMoversEmbeddings) is based on Python 2 and MATLAB.

## Prerequisites

Python 3 with the following additional packages
+ numpy
+ scipy
+ [dask](https://dask.org/)
+ [pyemb](https://pypi.org/project/pyemd/)
+ [gensim](https://radimrehurek.com/gensim/)


## How to use

Different from the original MATLAB implementation, which requires document labels to be provided and performs train-test splitting in the preprocessing steps, this implementation only requires the input of document text and produces the WME vectors for all of the input documents given a particular configuration of hyperparameters. 

The input document file should be a plain text file where each line corresponds to one document, and words in the document are separated by space. See `./data/twitter_docs.txt` for an example.

By default, the pre-trained word embedding matrix required by the WME algorithm is downloaded automatically via `gensim`. Check [Downloader API for gensim](https://radimrehurek.com/gensim/downloader.html) for supported options.

Example usage:
```
python wme.py ./data/twitter_docs.txt
```
which is equivalent with
```
python wme.py ./data/twitter_docs.txt --gamma 1.0 --R 128 --D_max 6 --D_min 1 --nuw_max 500 --wv_name word2vec-google-news-300 --exp_id exp_0
```

Check `python wme.py --help` for the meaning of the options. 

Suppose that `./data/twitter_docs.txt` is used as the corpus file and the `--exp_id` option is set to `exp_0`. Then the resulted WME document vectors will be saved as `./data/twitter_docs_wme_exp_0.npy`, which is a numpy array of which the i-th row stores the WME vector of the i-th document in the corpus file. It can be loaded by using the following commands.

```python
import numpy as np
wme_mat = np.load('./data/twitter_docs_wme_exp_0.npy')
```

The `num_workers` variable in script `wme.py` specifies the number of CPU cores that will be used for computing WME vectors in parallel. By default, a caching mechanism is activated so the user can abort the execution of the program anytime and later pick up based on what has been done last time. This mechanism will create a `tmp` folder in the project directory, and store a numpy array file whenever one WME vector is computed. In the previous example, the cache for the i-th document will be saved as `./tmp/exp_0/twitter_docs_i_wme.npy`. When the program is called, it will load the cached WME vector for a document if it exists; otherwise, it will compute the WME vector and then cache it. So please make sure to clean the `tmp` folder if you want to compute freshly new WME vectors.

Reproducibility can be achieved by specifying the `seed` variable in script `wme.py`.

## Remarks



## Comparison



## Reference

Wu, Lingfei, Ian En-Hsu Yen, Kun Xu, Fangli Xu, Avinash Balakrishnan, Pin-Yu Chen, Pradeep Ravikumar, and Michael J. Witbrock. 2018. “Word Mover’s Embedding: From Word2Vec to Document Embedding.” In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 4524–34.