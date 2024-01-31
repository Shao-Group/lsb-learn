Introduction
==============

- A learnable LSB function is an [LSB function](https://github.com/Shao-Group/lsbucketing) trained by machine learning methods:

- A Siamese neural network is used as a training framework in which the inner model is an inception neural network. The inception neural network merges a set of CNNs that adopt various sizes of kernels to capture kmer features.

- A loss function on sample $(s,t,y)$ in training set $T$: $L := \textstyle \sum_{(s,t,y)\in T}  L(s,t,y)$, where $y \in \lbrace1, -1\rbrace$ indicating whether $edit(s,t) \le d_1$, in which case $y = -1$, or whether $edit(s,t) \ge d_2$, in which case $y = 1$. Refer to the manuscript for more details.


Examples
==============
- Environment: python vision >= 3.6

- Data reading and uploading:
 
- Model Training:
The codes of all the functions (including losses, evaluations, and so on) and model structure are in `siacnn_models_gpu.py`. The `siaincp_runner.py` is a trainer for Siamese Neural Network. The parameters are easily modified in the code, as shown in the files. To train a model with the command:

`python siaincp_runner.py`

- Testing:
`tester.py` in `seq_n20/functions` is a small example of testing data `seq-n20-ED15-2.txt` for the trained models stored in `trained models` with the command:

`python tester.py`
