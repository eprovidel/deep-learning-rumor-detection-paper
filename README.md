# Using Deep Learning to Detect Rumors in Twitter

Paper URL: https://dl.acm.org/doi/10.1007/978-3-030-49570-1_22

## Abstract

The automatic detection of rumors in social networks is an important problem
that would allow counteracting the effects that the propagation of false
information produces. We study the performance of deep learning architectures
in this problem, analyzing ten different machines on word2vec and BERT. Our
results show that some architectures are more suitable for some particular
classes, suggesting that the use of committee machines would offer advantages
in this task.

## Datasets

In this paper we use the Twitter16 dataset as published by Wei Gao et al. in
[this link](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0).
The folder contains both Twitter15 and Twitter16 datasets. As specified by the
original authors: *"Note that constrained by the terms of Twitter service, we
cannot contain the content of the rest of the tweets. Data users can obtain the
sepcifics based on the provided tweet IDs and uids by their own."*

## Code

The code used to perform the study is available in the `W2Vmbedding_v2.ipynb`
notebook, which uses the `TweetModels.py` and `TweetUtils.py` libraries.
