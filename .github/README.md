Introduction
==============

A learnable LSB function is an [LSB function](https://github.com/Shao-Group/lsbucketing) trained by machine learning methods:

- A Siamese neural network is used as a training framework in which the inner model is an inception neural network. The inception neural network merges a set of CNNs that adopt various sizes of kernels to capture kmer features.

- A loss function on sample $(s,t,y)$ in training set $T$: $L := \textstyle \sum_{(s,t,y)\in T}  L(s,t,y)$, where $y \in \lbrace1, -1\rbrace$ indicating whether $edit(s,t) \le d_1$, in which case $y = -1$, or whether $edit(s,t) \ge d_2$, in which case $y = 1$. Refer to the manuscript for more details.


Examples
==============
- Environment: python vision >= 3.6

- Data simulating:
Codes in `/simulation` can generate a set of random pairs of length-n sequences $(s,t)$  with various edit distances as needed. 
 
- Model training:
`siacnn_models_gpu.py` is a function library (including losses, evaluations, model structures and generating hash code) awaiting import. The `siaincp_runner.py` is a trainer for Siamese Neural Network. Parameters are easily modified in the files following the annotations. To train a model with the command:

    `python siaincp_runner.py`

- Testing and hashcode generating:
[tester.py](https://github.com/Shao-Group/lsb-learn/blob/master/seq_n20/functions/tester.py) is a quick example of testing data `seq-n20-ED15-2.txt` for the trained models stored in `trained models` and generating the hash code with the command:

    `python tester.py`

Hash codes will be stored in a file named `hashcode_20k_40m_(d1,d2)s.hdf5`.
