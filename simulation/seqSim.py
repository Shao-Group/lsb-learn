import random
import string
import math

alphabet = ['A','T','C','G']

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return float(len(set1.intersection(set2)) / len(set1.union(set2)))

def minHash(s1, s2):
    #unweighted
    s = s1.union(s2)
    rp = random.sample(s, 1)
    common = s1.intersection(s2)
    #print(rp[0])
    if rp[0] in common:
        return 1
    return 0

def generate_random_sequence(length):
    #alphabet = ['A','T','C','G']
    sequence = ''.join(random.choice(alphabet) for i in range(length))
    #print("Sequence:", sequence)
    return sequence

def generate_sequence_similarity(input1, ratio):
    n = len(input1)
    if(ratio<1):
        ED = math.floor(n*(1-ratio))
    else:
         ED = ratio
    #print("ED:", ED)

    if math.floor(ED/2)>0:
        indel = random.randrange(math.floor(ED/2+0.01)+1)
        #print(ED/2)
    else:
        indel = 0
    sub = ED-2*indel
    #print("indel: ",indel,"sub: ", sub)

    input2 = input1
    for i in range(sub):
        posSub = random.randrange(n)
        ch = random.choice(alphabet)
        input2 = input2[:posSub]+ch+input2[(posSub+1):]
        #print(posSub, ch, input2)

    for i in range(indel):
        posInsert = random.randrange(n)
        posDelete = random.randrange(n)
        ch = random.choice(alphabet)
        #print(posInsert, posDelete, ch)
        input2 = input2[:posInsert]+ch+input2[posInsert:]
        input2 = input2[:posDelete]+input2[(posDelete+1):]

    #print(input1, input2)
    return input2

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [ [ 0 for i in range(size_y) ] for j in range(size_x) ]
    #matrix = numpy.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )
    #print(matrix)
    return (matrix[size_x - 1][size_y - 1])

#print(levenshtein("AAATGTG", "AACTGTG"))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--l', '--length', type=int, default=100, help="Length of sequence")
parser.add_argument('--s', '--size', type=int, default=100, help="Num of seqs for each d")
parser.add_argument('--d', '--dist', type=int, default=20, help="Max edit distance")
parser.add_argument('--r', '--run', type=int, default=0, help="ID of run")
args = parser.parse_args()

if  __name__ == "__main__":
    l = args.l
    maxED = args.d
    dSize = args.s
    run = args.r
    i = 0
    countED = [ 0 for i in range(l+1)]
    dataID = 'seq-n' + str(l) + '-ED' + str(maxED) 
    f = open('/data/qzs23/projects/seqML/'+dataID +'/'+dataID+'-' + str(run)+'.txt', 'a')
    totalSize = dSize * maxED;
    while(sum(countED[1:maxED+1])< totalSize):
        i = i+1
        for ratio in range(1,maxED+30):
            S1 = generate_random_sequence(l)
            S2 = generate_sequence_similarity(S1,ratio) 
            ED = levenshtein(S1, S2)
            if(ED == 0): continue
            #if(ED > maxED ): continue
            if(countED[ED] >= dSize): continue
            countED[ED] += 1
            print(S1, S2, ED, file=f)
        if(i % 1000 == 0): 
            print(countED)
    print(countED)
    f.close()
