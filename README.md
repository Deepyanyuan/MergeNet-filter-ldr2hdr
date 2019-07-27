# MergeNet-filter-ldr2hdr
MergeNet-filter-ldr2hdr, detail in paper 《Reconstructing HDR Image from a Single Filtered LDR Image Base on a Deep HDR Merger Network》 

original HDR dataset come from online: DML-HDR {http://dml.ece.ubc.ca/data/DML-HDR/} and Funt-HDR {https://www2.cs.sfu.ca/~colour/data/}.

step 0: generate training pairs by generate_data.py or generate_data_2.py

step 1: create network, in this paper, we use Endo et al.'s network and make a small modification, detail in network.py and our paper.

step 2: train this network by train_network.py. Note that in this paper, we use filtered LDR images as input, and traditional LDR images and log-domain HDR image as ground-truth, detail in our paper.

step 3: test this network by test_network.py.

step 4: predict the results if input any LDR image by main.py

step 5: performance comparison by Matlab code
