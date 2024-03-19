/*
 * Given two files and a threshold t, use the WFA2-lib to compute pairwise
 * edit distance between the two files with the band and max_step both set to
 * t+1. Output pairs that have edit distances <= t.
 *
 * Based on WFA2-lib/examples/wfa_binding.cpp.
 *
 * By: Ke@PSU
 * Last edited: 03/18/2024
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "bindings/cpp/WFAligner.hpp"

using namespace std;
using namespace wfa;

// output t if edit(s,t) < threshold
inline bool testPair(WFAlignerEdit& aligner,
		     string& s, string& t,
		     const int threshold){
    aligner.alignEnd2End(s, t);
    return (aligner.getAlignmentStatus() != WFAligner::StatusMaxStepsReached
	&& aligner.getAlignmentScore() < threshold);
}

int main(int argc,char* argv[]) {
    if(argc != 5){
	fprintf(stderr, "Usage: pairwise_ed.out file1 file2 threshold output_file\n");
	return 1;
    }

    int threshold = atoi(argv[3]) + 1;
    struct stat test_file;
    int i;
    for(i=1; i<2; ++i){
	if(stat(argv[i], &test_file) != 0){
	    fprintf(stderr, "Cannot read file %s\n", argv[i]);
	    return 1;
	}
    }

    ifstream fin1(argv[1]);
    vector<string> seq_list;
    string seq;
    while(getline(fin1, seq)){
	seq_list.push_back(seq);
    }

    ifstream fin2;
    vector<string> seq_list2;
    bool same_list = true;
    if(strcmp(argv[1], argv[2]) != 0){
	same_list = false;
	fin2.open(argv[2]);
	while(getline(fin2, seq)){
	    seq_list2.push_back(seq);
	}
    }

    ofstream fout(argv[4]);
    // Patter & Text
    // string pattern = "TCTTTACTCGCGCGTTGGAGAAATACAATAGT";
    // string text    = "TCTATACTGCGCGTTTGGAGAAATAAAATAGT";

    // Create a WFAligner
    WFAlignerEdit aligner(WFAligner::Score);
    aligner.setHeuristicBandedStatic(-threshold, threshold);
    aligner.setMaxAlignmentSteps(threshold);

    // compute edit distance for each seq in file2 against all in whitelist
    if(!same_list){
	for(string& s : seq_list2){
	    fout << s;
	    for(string& t : seq_list){
		if(testPair(aligner, s, t, threshold)){
		    fout<< " " << t;
		}
	    }
	    fout << endl;
	}
    }else{
	for(int i=0; i<seq_list.size(); ++i){
	    fout << seq_list[i];
	    for(int j=i+1; j<seq_list.size(); ++j){
		if(testPair(aligner, seq_list[i], seq_list[j], threshold)){
		    fout<< " " << seq_list[j];
		}
	    }
	    fout << endl;
	}
    }

    // Print CIGAR
    // string cigar = aligner.getAlignment();
    // cout << "PATTERN: " << pattern  << endl;
    // cout << "TEXT: " << text  << endl;
    // cout << "CIGAR: " << cigar  << endl;
    return 0;
}
