# MergeNet-filter-ldr2hdr
MergeNet-filter-ldr2hdr, detail in paper 《Reconstructing HDR Image from a Single Filtered LDR Image Base on a Deep HDR Merger Network》 



2019-7-1

pipeline:

original HDR dataset come from online: DML-HDR {http://dml.ece.ubc.ca/data/DML-HDR/} and Funt-HDR {https://www2.cs.sfu.ca/~colour/data/}.

step 0: generate training pairs by generate_data.py or generate_data_2.py

step 1: create network, in this paper, we use Endo et al.'s network and make a small modification, detail in network.py and our paper.

step 2: train this network by train_network.py. Note that in this paper, we use filtered LDR images as input, and traditional LDR images and log-domain HDR image as ground-truth, detail in our paper.

step 3: test this network by test_network.py.

step 4: predict the results if input any LDR image by main.py

step 5: performance comparison by Matlab code


2019-9-01

supplement


Added PDF version of the paper, see ISMAR2019-BinLiang.pdf

supplement

Added a complete datasets and pre-trained parameters, linked as follows

linked：https://pan.baidu.com/s/1dD5NMOV8Cov3G_F5WHjwTw 
Extraction code：ccwk 

Note that after downloading, you need to extract all the compressed files first, the default path can be
