#pragma once
#include "Parameters.h"
#include "RNN.h"


class RNNMaster
{
public:
	Parameters *parameters;
	//put frequently used variables here
	int batchNum;
	string rnnModelFile;
	string trainFile;
	int* wordCounts;

	RNNMaster(Parameters &parameters);

	~RNNMaster() {
		/*if(parameters != NULL) {
			free(parameters);
		}*/
		if(wordCounts != NULL) {
			free(wordCounts);
		}
	}


	void partitionAndDispatch(Vocab *vocab, int iteration, bool use_hpc, FILE *logger);
	void saveMasterRNNModel(RNN &model);
	void dispatchSlaveTrain(int iteration, int batchId, bool use_hpc, FILE* logger);
	bool finisedSlaveRnnsTrain(int iteration);
	RNN* averageRnnModels(RNN &preAveModel, FILE* logger, double tolerance, bool master_para_adapt, double learningRate, double beta);
	static void master(string paraFile);
	static void slave(string rnnFile, string batchTrainFile, int batchWordCount, string batchRnnFile);

	static void masterTemp(string paraFile);
    static void masterTemp2(string paraFile);
};

