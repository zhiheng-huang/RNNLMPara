# RNNLMPara
Parallel RNN trainer implementes of the following paper

Z. H.  Huang, G. Zweig, M. Levit, B. Dumoulin, B. Oguz and S. Chang, Accelerating Recurrent Neural 
Network Training via Two Stage Classes and Parallelization, in Automatic Speech Recognition and 
Understanding (ASRU), 2013.

RNN parallel training by dispatching multiple HPC jobs for slave RNN training. Code is developed 
based on Recurrent neural network based statistical language modeling toolkit Version 0.3f (Tomas 
Mikolov). The following changes are made

1) Separate Vocab part to a class

2) Add two stage class (super class and class) to speed up training. Two options to generate 
super classes: even or frequency based.

3) new Maxent feature hash function

4) Explicit RNN constructors from random initialization or from model file

5) Submit HPC jobs to train slave RNN models

6) Master model update after done with HPC slave RNN model training

See RNNOrigExp/runPennTreebank.sh and RNNParaExp/readme.txt for experiments. 




