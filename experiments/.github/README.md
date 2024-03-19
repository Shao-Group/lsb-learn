### Code and scripts for reproducing experimental results

- Comparison with Order Min Hash (OMH):
	- OMH sketches are computed with `omh_sketch` of [omh_compute-0.0.2](https://github.com/Kingsford-Group/omhismb2019/releases/tag/v0.0.2). To accommodate its input format, `N` pairs of length-`n` strings with edit distance `d` are stored in a fasta file with `2N` records; records labeled `>x_1` and `>x_2` form a pair, see [20mers-ed3.fa](../example_data/20mers-ed3.fa) for an example.
	- To generate hash codes with OMH, run `omh_sketch -k5 -l2 -m20 -o 20mers-ed3.omh-out 20mers-ed3.fa`.
	- To compute number of pairs that are assigned to a same bucket according to the OMH sketches, run `python countOMHCollisions.py 20mers-ed3.omh-out`.
- Comparison with WFA:
	- The [WFA2-lib](https://github.com/smarco/WFA2-lib) library is used to compare the running time with WFA on the barcode experiment.
	- Once WFA2-lib is compiled, [pairwise_ed.cpp](../pairwise_ed.cpp) can be compiled (assuming it is inside example/ of the WFA2-lib directory) with `g++  -L../lib -I.. pairwise_ed.cpp -o bin/pairwise_ed.out -lwfacpp -fopenmp -lm`.
	- The program takes two files as input and compute pairwise edit distances by WFA between the sequences in the two files. Pairs that have an edit distance at most the provided threshold are output to a file. Two sample input files are provided in [example_data](../example_data).
	- For the barcode experiment, run `pairwise_ed.out example_data/whitelist.txt example_data/mismatches.txt 2 output.txt`.
	- For the larger-scale pairwise comparison between the mismatched barcodes, run `pairwise_ed.out example_data/mismatches.txt example_data/mismatches.txt 2 output.txt`.
