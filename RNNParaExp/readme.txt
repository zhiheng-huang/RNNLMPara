-------------------------------------------------------------------------------
1. Parallel RNN training 
-------------------------------------------------------------------------------

RNN parallel training by dispatching multiple HPC jobs for slave RNN training. Code is developed based on Recurrent neural network based statistical language modeling toolkit Version 0.3f (Tomas Mikolov). The following changes are made

1) Separate Vocab part to a class
2) Add two stage class (super class and class) to speed up training. Two options to generate super classes: even or frequency based.
2) new Maxent feature hash function
2) Explicit RNN constructors from random initialization or from model file
3) Submit HPC jobs to train slave RNN models
4) Master model update after done with HPC slave RNN model training

The following command runs the parallel RNN training

job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM RNNLM.exe master configFile

See data/ConfigIspeech100KBatch for an example configure file. All parameters includes training/valid/test data files, batch size, learning rate etc are specified in configure file.

job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM /memorypernode:10000 /jobtemplate:Express /jobname:ptb \\fbl\nas\HOME\zhihuang\work\RNNLM\x64\Release\RNNLM.exe master \\fbl\nas\home\zhihuang\work\RNNParaExp\ConfigPennTreebank
