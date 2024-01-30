Introduction
==============

A learnable LSB function is a LSB function trained by machine learning methods:

- A $(d_1,d_2)$-LSB function sends sequences into multiple buckets with the guarantee that pairs of sequences of edit distance at most $d_1$ can be found within the same bucket while those of edit distance at least $d_2$ do not share any.

- A Siamese neural network is used as a training framework in which the inner model is an inception neural network. The inception neural network merges a set of CNNs that adopt various sizes of kernels to capture kmer features.

- A loss function on sample $(s,t,y)$: $L := \textstyle \sum \lbrace(s,t,y)\in T \rbrace$, where $y \in \lbrace1, -1\rbrace$ indicating whether $edit(s,t) \le d_1$, in which case $y = -1$, or whether $edit(s,t) \ge d_2$, in which case $y = 1$. Refer to the manuscript for more details.

Examples
==============

The codes of all the functions (including losses, evaluations, and so on) and model structure are in `siacnn_models_gpu.py`. The `siaincp_runner.py` is a trainer for Siamese Neural Network.

`tester.py` in `seq_n20/functions` is a small example of testing data `seq-n20-ED15-2.txt` for the trained models stored in `trained models`.
