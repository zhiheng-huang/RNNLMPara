# RNNLMPara
## Why this tool?

Parallel RNN trainer implementes Two stage class RNNs and parallel RNNs proposed in the following paper

```bash
Z. H.  Huang, G. Zweig, M. Levit, B. Dumoulin, B. Oguz and S. Chang, Accelerating Recurrent Neural 
Network Training via Two Stage Classes and Parallelization, in Automatic Speech Recognition and 
Understanding (ASRU), 2013.
```

Two stage class RNNs uses two stage classes (super classes and classes) as opposed to one class. Parallel RNN trainer splits the training data into batches and then dispatchs jobs to multiple CPUs/nodes for slave models training. Two stage class RNNs and parallel RNNs not only result in equal or lower WERs compared to original RNNs but also accelerate training by 2 and 10 times respectively. Code is developed based on [RNNLM 0.3e (Tomas Mikolov)](http://www.fit.vutbr.cz/~imikolov/rnnlm/). The following changes are made

  1) Separate Vocab part to a class

  2) Add two stage class (super class and class) to speed up training. Two options to generate 
  super classes: even or frequency based.

  3) new Maxent feature hash function

  4) Explicit RNN constructors from random initialization or from model file

  5) Submit HPC jobs to train slave RNN models

  6) Master model update after done with HPC slave RNN model training

## Usage

### Build
To build, run `build.sh` to generate the binary at `Release/RNNLMPara`

### Experiments
See `RNNOrigExp/runPennTreebank.sh` and `RNNParaExp/readme.txt` for experiments. 

## Contact
Please send your questions/comments to zhiheng.huang@gmail.com

