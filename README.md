Introduction
==============

This repo hosts source code to train locality-sensitive bucketing (LSB) functions.
A bucketing function $f$ maps a length $n$ string to a set of hash-codes (instead of one hash-code).
A bucketing function $f$ is said to be $(d_1, d_2)$-sensitive,
if for any two length-n strings $s$ and $t$, $f$ satisfies: if the edit distance between $s$ and $t$ 
is less than or equal to $d_1$, then $f(s)$ and $f(t)$ share at least one hash-code,
and if the edit distance between $s$ and $t$ 
is greater than or equal to $d_2$, then $f(s)$ and $f(t)$ will not share any hash-code.
The LSB functions are proposed in [LSB paper](https://doi.org/10.1186/s13015-023-00234-2)
with source code at [LSB repo](https://github.com/Shao-Group/lsbucketing).

Here we develop a machine-learning framework to automatically learn $(d_1, d_2)$-LSB functions
from simulation data.  Briefly speaking, we use Siamese neural network as the training framework 
in which the inner model is an inception neural network which represents the hash function $f$. 
The inception neural network consists of layers of convolution-maxpooling units
that can capture various sizes of substrings as features.


Usage
==============
- Environment: python vision >= 3.6

- Data simulation.
Codes in `/simulation` can generate a set of random pairs of length-n strings
$(s,t)$  with various edit distances as needed. 
Given $d_1, d_2$, training samples consist of tuples $\{(s,t,y)\}$,
$y = -1$ if $edit(s,t) \le d_1$ and $y = 1$ if $edit(s,t) \ge d_2$.

- Model training. Codes for $n = 20$ and $n=100$ are put in separate folders.
`siacnn_models_gpu.py` is a function library (including losses, evaluations,
model structures and generating hash code) awaiting import. The
`siaincp_runner.py` is a trainer for Siamese Neural Network. Parameters are
easily modified in the files following the annotations. To train a model, 
use command:

    `python siaincp_runner.py`

- Testing and hashcode generating.
[tester.py](https://github.com/Shao-Group/lsb-learn/blob/master/seq_n20/functions/tester.py)
is a quick example of testing data `seq-n20-ED15-2.txt` for the pretained models
stored in `trained models` and generating the hash code with the command;
hash codes will be stored in a file named `hashcode_20k_40m_(d1,d2)s.hdf5`.

    `python tester.py`

- Pre-trained models. More pre-trained models are available at 
[zenodo](https://zenodo.org/records/10655349).
