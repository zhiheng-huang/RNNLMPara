#include "RNNExp.h"
#include <string>
#include "Parameters.h"
#include "RNN.h"
#include "Utils.h"
#include "RNNMaster.h"

using namespace std;

void RNNExp::standaloneRun(string paraFile) 
{
	Parameters parameters(paraFile);
	bool generate_interactive = false;
	int head = -1;

	if (parameters.getParaBool("train_model"))
	{
		RNN *model1 = NULL;	
		string rnnModelFile = parameters.getPara("rnnlm_file");
		string logFile = rnnModelFile + "Log";
		ifstream file(rnnModelFile);
		if (file.good())
		{
			model1 = new RNN(rnnModelFile, true);
			FILE *logger=fopen(logFile.c_str(), "ab");
			model1->evaluateNet(logger);
			model1->netReset(); //zhiheng, fix the divergence bug??
		} else {
			bool starRandom = parameters.getParaBool("random_start");
			model1 = new RNN(parameters, starRandom);
		}
		model1->trainNet(false, logFile);
		model1->saveNet(rnnModelFile, true);
		delete model1;
	}

	if (parameters.getParaBool("test_model"))
	{
		RNN model1(parameters.getPara("rnnlm_file"), true);
		int debug_mode = parameters.getParaInt("debug_mode", 0);
		string logName = parameters.getPara("rnnlm_file") + "_pplx";
		//FILE *logger=fopen(logName.c_str(), "ab");
		FILE *logger=fopen(logName.c_str(), "w");
		model1.testNet(parameters.getPara("test_file"), parameters.getParaBool("replace"), parameters.getParaDouble("unk_penalty", 0), logger, debug_mode);
	}

	int gen = parameters.getParaInt("gen", 0);
	if (gen > 0)
	{
		RNN model1(parameters.getPara("rnnlm_file"), true);
		string outFile = parameters.getPara("rnnlm_file") + "_gen";
		model1.testGen(gen, head, generate_interactive, outFile);
	}

	if (parameters.getParaInt("savewp", 0) > 0)
	{
		RNN model1(parameters.getPara("rnnlm_file"), true);
		//model1.setRandSeed(rand_seed);
		model1.saveWordProjections();
	}

	//if (lattice_file != "") {
	//	CRnnLM model1;
	//	model1.setRnnLMFile(rnnlm_file);
	//	if (fea_file_set==1) {
	//		model1.setFeaFile(fea_file);
	//		model1.setFeaSize(fea_size);
	//	}
	//	model1.setDebugMode(debug_mode);
	//	model1.setUnkPenalty(unk_penalty);

	//	vector<string> ef;
	//	vector<float> ew, lw;
	//	if (external_weights != "") {
	//		if (external_format == "") {
	//			cerr << "An external format must be specified when external weights are specified\n";
	//			exit(1);
	//		}
	//		parse_external_weights(external_weights, ew);
	//	}
	//	if (external_format != "") {
	//		parse_external_format(external_format, ef); //zhiheng, am=,lm=,pen= -> <am=,lm=,pen=>
	//	}
	//	if (linear_weights != "") {
	//		if (external_format == "") {
	//			cerr << "An external format must be specified when linear-interpolation is specified\n";
	//			exit(1);
	//		}
	//		parse_external_weights(linear_weights, lw);
	//		assert(lw.size() > 2);  // at least a RNN weight, and logarithmic weight plus one other
	//		check_linear_weights(lw);
	//		cout << "Setting weights for linear interpolation. RNN will get: " << lw[lw.size()-2] << endl;
	//		rnn_weight = lw[lw.size()-2];
	//	}

	//	rescorer R(&model1, ew, ef, lw, nbest, rnn_weight);
	//	R.rescore(lattice_file.c_str(), ((context_file=="")?(NULL):(context_file.c_str())));
	//} //zhiheng, end if (lattice_file != "") 
}

int main(int argc, char **argv)
{
	//string paraFile = "//fbl/nas/HOME/zhihuang/RNNParaExp/ConfigIspeech100KBatch";
	//string paraFile = "//fbl/nas/HOME/zhihuang/RNNParaExp/ConfigIspeech5KBatch";

	//string modelFile = "//fbl/nas/HOME/zhihuang/RNNParaExp/google99-17/modelRnn";
	//string modelFile2 = modelFile + "Modified";

	//RNN* model = new RNN(modelFile, true); //do not read vocab info
	//model->alpha = 0.05;
	//model->saveNet(modelFile2, true);
	//return 0;
	string str = argv[1];
	Utils::lowercase(str);
	if(str.compare("master") == 0) {
		RNNMaster::master(argv[2]);
	} else if(str.compare("slave") == 0) {
		RNNMaster::slave(argv[2], argv[3], atoi(argv[4]), argv[5]);
	} else if(str.compare("standalone") == 0) {
		RNNExp::standaloneRun(argv[2]);
	} else if(str.compare("test") == 0) { //used for test stage
		string rnnModel = "";
		string testFile = "";
		string outFile = "";
		bool replace = false;
		double unk_penalty = 0;
		string str;
		for(int i = 2; i < argc; i = i+2) {
			string s = argv[i]; 
			if(s.compare("-rnnlm") == 0) {
				rnnModel = argv[i+1];
			} else if(s.compare("-test") == 0) {
				testFile = argv[i+1];
			} else if(s.compare("-output") == 0) {
				outFile = argv[i+1];
			} else if(s.compare("-replace") == 0) {
				str = argv[i+1];
				if(str.compare("true") == 0) {
					replace = true;
				}
			} else if(s.compare("-unk_penalty") == 0) {
				str = argv[i+1];
				unk_penalty = Utils::str2Double(str);
			}
		}
		if(rnnModel.length() == 0 || testFile.length() == 0 || outFile.length() == 0) {
			printf("model file, test file or output file not specified\n");
			exit(1);
		}
		RNN model(rnnModel, true);
		model.independent = true;
		int debug_mode = 2;		
		FILE *logger=fopen(outFile.c_str(), "w");
		model.testNet(testFile, replace, unk_penalty, logger, debug_mode);
	} else {
		cerr << "Usage: RNNLMPara.exe master paraFile" << endl;
		cerr << "Usage: RNNLMPara.exe slave rnnModel batchTrainFile batchWordCount batchRnnFile" << endl;
		cerr << "Usage: RNNLMPara.exe standalone paraFile" << endl;
		exit(1);
	}

	/*string paraFile = "//fbl/nas/HOME/zhihuang/RNNParaExp/ConfigIspeech10MBatch";
	RNNMaster::masterTemp(paraFile);*/
	//return 0;
}
