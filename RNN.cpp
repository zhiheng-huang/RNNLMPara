#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sstream>
#include "RNN.h"
#include "Utils.h"
#include <locale> //string to long hash

///// include blas
#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif
//

using namespace std;

real RNN::exp_10(real num) {return exp(num*2.302585093);}		//in VS, exp10() is not defined...

//Initial RNN model constructor from a parameter file
RNN::RNN(Parameters parameters, bool useRandom)
{	
	version = 0;
	train_file = parameters.getPara("train_file");
	valid_file = parameters.getPara("valid_file");
	feature_gamma = 0.9;
	rand_seed = parameters.getParaInt("random_seed", 1); //2
	srand(rand_seed);
	random_start = parameters.getParaBool("random_start");
	file_binary = parameters.getParaBool("file_binary");
	string s = parameters.getPara("context_ids");
	if(s.length() > 0) {
		vector<string> ids;
		Utils::Tokenize(s, ids, ":");
		for(int i = 0; i < ids.size(); i++) {
			int id = atoi(ids.at(i).c_str());
			if(id >= 0) {
				context_ids.push_back(id);
			}
		}
	}

	//to add later
	//if (fea_file_set==1) {
	//    model1.setFeaFile(fea_file);
	//    model1.setFeaSize(fea_size);
	//    if (fea_valid_file_set==1) {
	//        model1.setFeaValidFile(fea_valid_file);
	//    } else {
	//        printf("ERROR: For training with features, validation feature file must be specified using -features-valid <file>\n");
	//        exit(1);
	//    }
	//}

	//if (fea_matrix_file_set==1) {
	//    model1.setFeaMatrixFile(fea_matrix_file);
	//    model1.setFeatureGamma(feature_gamma);
	//}

	string class_file = parameters.getPara("class_file");
	vocab = new Vocab(parameters.getPara("train_file"));
	if (!class_file.empty())
	{
		vocab->sortVocabByClass(class_file);
	}
	else
	{
		vocab->sortVocabByFreq(parameters.getParaInt("class_size", 10), parameters.getParaBool("old_classes"), 
			parameters.getParaInt("super_class_size", -1), parameters.getParaBool("super_class_even"));
	}

	gradient_cutoff = parameters.getParaDouble("gradient_cutoff", 15);
	starting_alpha = parameters.getParaDouble("starting_alpha", 0.1);
	alpha = parameters.getParaDouble("starting_alpha", 0.1);
	alpha_divide = false;

	llogpTrain = -100000000;
	logpTrain = -100000000;
	llogpValid = -100000000;
	logpValid = -100000000;

	min_improvement = parameters.getParaDouble("min_improvement", 1.0002);
	iter = 0;
	one_iter = false;
	maxIter = parameters.getParaInt("maxIter", 0);
	train_words = vocab->getWordCount();
	train_cur_pos = 0;
	counter = vocab->getWordCount();
	beta = parameters.getParaDouble("regularization", 0.0000001);

	layer1_size = parameters.getParaInt("hidden_size", 100);
	layer0_size=vocab->vocabSize()+layer1_size;
	layer2_size=vocab->vocabSize()+vocab->classSize;
	if(vocab->superClassSize > 1) {
		layer2_size += vocab->superClassSize;
	}
	layerc_size = parameters.getParaInt("compression_size", 0);
	direct_size = parameters.getParaInt("direct", 0); //int to long long
	direct_order = parameters.getParaInt("direct_order", 3);
	direct_word_size = parameters.getParaInt("direct_word_size", 0);
	direct_class_size = parameters.getParaInt("direct_class_size", 0);
	bptt = parameters.getParaInt("bptt", 5);
	bptt_block = parameters.getParaInt("bptt_block", 10);
	fea_size = 0;
	fea_matrix_used = false;
	independent = parameters.getParaBool("independent");
	initNet(useRandom);
	//featureIndexer = NULL;
	//featureIndexer = new FeatureIndexer(parameters.getPara("train_file"), vocab, 3);
	//featureIndexer->toString("c:/temp/features");
}

//Constructor via reading a RNN model from a given model file
RNN::RNN(string rnnlm_file, bool includeVocab)
{		
	ifstream fi(rnnlm_file, ios::in | ios::binary);	
	if(!fi.good()) {
		cerr << "model file " << rnnlm_file << " is not existing" << endl;
		exit(1);
	}
	float fl;
	int a, b;
	string line;

	version = atoi(Utils::getVal(fi).c_str());
	rand_seed = atoi(Utils::getVal(fi).c_str());
	srand(rand_seed);
	string s = Utils::getVal(fi);
	if(s.length() > 0) {
		vector<string> ids;
		Utils::Tokenize(s, ids, ":");
		for(int i = 0; i < ids.size(); i++) {
			int id = atoi(ids.at(i).c_str());
			if(id >= 0) {
				context_ids.push_back(id);
			}
		}
	}
	file_binary = Utils::str2Bool(Utils::getVal(fi));
	train_file = Utils::getVal(fi);
	valid_file = Utils::getVal(fi);
	llogpValid = Utils::str2Double(Utils::getVal(fi));
	logpValid = Utils::str2Double(Utils::getVal(fi));

	iter = atoi(Utils::getVal(fi).c_str());
	train_cur_pos = atoi(Utils::getVal(fi).c_str());
	llogpTrain = Utils::str2Double(Utils::getVal(fi));
	logpTrain = Utils::str2Double(Utils::getVal(fi));
	train_words = atoi(Utils::getVal(fi).c_str());
	layer0_size = atoi(Utils::getVal(fi).c_str());
	fea_size = atoi(Utils::getVal(fi).c_str());
	fea_matrix_used = Utils::str2Bool(Utils::getVal(fi));
	feature_gamma = Utils::str2Double(Utils::getVal(fi));

	layer1_size = atoi(Utils::getVal(fi).c_str());
	layerc_size = atoi(Utils::getVal(fi).c_str());
	layer2_size = atoi(Utils::getVal(fi).c_str());
	direct_size = atoi(Utils::getVal(fi).c_str());
	direct_order = atoi(Utils::getVal(fi).c_str());
	direct_word_size = atoi(Utils::getVal(fi).c_str());
	direct_class_size = atoi(Utils::getVal(fi).c_str());
	bptt = atoi(Utils::getVal(fi).c_str());
	bptt_block = atoi(Utils::getVal(fi).c_str());
	independent = Utils::str2Bool(Utils::getVal(fi));

	starting_alpha = Utils::str2Double(Utils::getVal(fi));
	alpha = Utils::str2Double(Utils::getVal(fi));
	alpha_divide = Utils::str2Bool(Utils::getVal(fi));
	beta = Utils::str2Double(Utils::getVal(fi));
	int vocab_size = atoi(Utils::getVal(fi).c_str());
	//featureIndexer = new FeatureIndexer(featureFile);

	if(includeVocab) {
		vocab = new Vocab(fi);
	} else {
		vocab = NULL;
		getline(fi, line);
		assert(line.empty());
	}

	initNet(false);

	if (file_binary == false) {
		getline(fi, line); //Hidden layer activation:
		for (a=0; a<layer1_size; a++) {
			getline(fi, line);
			neu1[a].ac=Utils::str2Double(line);
		}
	}
	if (file_binary == true) {		
		for (a=0; a<layer1_size; a++) {
			fi.read((char*)&fl, sizeof(float));
			neu1[a].ac=fl;
		}
	}

	if (file_binary == false) {
		Utils::getVal(fi);
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<layer0_size; a++) {
				getline(fi, line);
				syn0[a+b*layer0_size].weight=Utils::str2Double(line);
			}
		}
	}
	if (file_binary == true) {
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<layer0_size; a++) {
				fi.read((char*)&fl, sizeof(float));
				syn0[a+b*layer0_size].weight=fl;
			}
		}
	}
	//
	if (file_binary == false) {
		Utils::getVal(fi);
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<fea_size; a++) {
				getline(fi, line);
				synf[a+b*fea_size].weight=Utils::str2Double(line);
			}
		}
		//
		Utils::getVal(fi);
		for (b=0; b<layer2_size; b++) {
			for (a=0; a<fea_size; a++) {
				getline(fi, line);
				synfo[a+b*fea_size].weight=Utils::str2Double(line);
			}
		}
	}
	if (file_binary == true) {
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<fea_size; a++) {
				fi.read((char*)&fl, sizeof(float));
				synf[a+b*fea_size].weight=fl;
			}
		}
		//
		for (b=0; b<layer2_size; b++) {
			for (a=0; a<fea_size; a++) {
				fi.read((char*)&fl, sizeof(float));
				synfo[a+b*fea_size].weight=fl;
			}
		}
	}
	////
	if (file_binary == false) {
		Utils::getVal(fi);
		if (layerc_size==0) {	//no compress layer
			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layer1_size; a++) {
					getline(fi, line);
					syn1[a+b*layer1_size].weight=Utils::str2Double(line);
				}
			}
		}
		else
		{				//with compress layer
			for (b=0; b<layerc_size; b++) {
				for (a=0; a<layer1_size; a++) {
					getline(fi, line);
					syn1[a+b*layer1_size].weight=Utils::str2Double(line);
				}
			}

			Utils::getVal(fi);

			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layerc_size; a++) {
					getline(fi, line);
					sync[a+b*layerc_size].weight=Utils::str2Double(line);
				}
			}
		}
	}
	if (file_binary == true) {
		if (layerc_size==0) {	//no compress layer
			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layer1_size; a++) {
					fi.read((char*)&fl, sizeof(float));
					syn1[a+b*layer1_size].weight=fl;
				}
			}
		}
		else
		{				//with compress layer
			for (b=0; b<layerc_size; b++) {
				for (a=0; a<layer1_size; a++) {
					fi.read((char*)&fl, sizeof(float));
					syn1[a+b*layer1_size].weight=fl;
				}
			}

			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layerc_size; a++) {
					fi.read((char*)&fl, sizeof(float));
					sync[a+b*layerc_size].weight=fl;
				}
			}
		}
	}
	////
	if (file_binary == false) {
		Utils::getVal(fi);		//direct conenctions
		long long aa;
		for (aa=0; aa<direct_size; aa++) {
			getline(fi, line);
			syn_d[aa]=Utils::str2Double(line);
		}
	}
	//
	if (file_binary == true) {
		long long aa;
		for (aa=0; aa<direct_size; aa++) {
			fi.read((char*)&fl, sizeof(float));
			syn_d[aa]=fl;
		}
	}

	/*if (fea_matrix_used) {
	if (fea_matrix==NULL) fea_matrix=(real *)calloc(vocab->vocabSize()*fea_size, sizeof(real));
	if (file_binary == false) {
	Utils::goToDelimiter(':', fi);
	for (b=0; b<vocab->vocabSize(); b++) {
	for (a=0; a<fea_size; a++) {
	fscanf(fi, "%lf", &d);
	fea_matrix[a+b*fea_size]=d;
	}
	}
	}
	if (file_binary == true) {
	for (b=0; b<vocab->vocabSize(); b++) {
	for (a=0; a<fea_size; a++) {
	fread(&fl, 4, 1, fi);
	fea_matrix[a+b*fea_size]=fl;
	}
	}
	}
	}*/
	//
	//saveWeights();
	fi.close();
}

RNN::RNN(RNN &model, bool includeVocab)
{	
	copyHead(model, includeVocab);
	copyNet(model);
	//featureIndexer = NULL; //to do
}

//copy head info from a given RNN model
void RNN::copyHead(RNN &model, bool includeVocab)
{
	version = model.version;
	rand_seed = model.rand_seed;
	srand(rand_seed);
	file_binary = model.file_binary;
	train_file = model.train_file;
	valid_file = model.valid_file;
	llogpValid = model.llogpValid;
	logpValid = model.logpValid;

	iter = model.iter;
	train_cur_pos = model.train_cur_pos;
	llogpTrain = model.llogpTrain;
	logpTrain = model.logpTrain;
	train_words = model.train_words;
	layer0_size = model.layer0_size;
	fea_size = model.fea_size;
	fea_matrix_used = model.fea_matrix_used;
	feature_gamma = model.feature_gamma;
	layer1_size = model.layer1_size;
	layerc_size = model.layerc_size;
	layer2_size = model.layer2_size;
	direct_size = model.direct_size;
	direct_order = model.direct_order;
	direct_word_size = model.direct_word_size;
	direct_class_size = model.direct_class_size;
	bptt = model.bptt;
	bptt_block = model.bptt_block;
	independent = model.independent;
	starting_alpha = model.starting_alpha;
	alpha = model.alpha;
	alpha_divide = model.alpha_divide;
	beta = model.beta;
	if(includeVocab) {
		vocab = new Vocab(*model.vocab);
	} else {
		vocab = NULL;
	}
}

//copy RNN net weights from a given RNN model
void RNN::copyNet(RNN &model)
{
	initNet(false); //allocate memory
	for (int a = 0; a < layer1_size; a++)
	{
		neu1[a].ac = model.neu1[a].ac;
	}

	for (int b = 0; b < layer1_size; b++)
	{
		for (int a = 0; a < layer0_size; a++)
		{
			syn0[a + b * layer0_size].weight = model.syn0[a + b * layer0_size].weight;
		}
	}


	for (int b = 0; b < layer1_size; b++)
	{
		for (int a = 0; a < fea_size; a++)
		{
			synf[a + b * fea_size].weight = model.synf[a + b * fea_size].weight;
		}
	}

	for (int b = 0; b < layer2_size; b++)
	{
		for (int a = 0; a < fea_size; a++)
		{
			synfo[a + b * fea_size].weight = model.synfo[a + b * fea_size].weight;
		}
	}

	if (layerc_size == 0)
	{	//no compress layer
		for (int b = 0; b < layer2_size; b++)
		{
			for (int a = 0; a < layer1_size; a++)
			{
				syn1[a + b * layer1_size].weight = model.syn1[a + b * layer1_size].weight;
			}
		}
	}
	else
	{				//with compress layer
		for (int b = 0; b < layerc_size; b++)
		{
			for (int a = 0; a < layer1_size; a++)
			{
				syn1[a + b * layer1_size].weight = model.syn1[a + b * layer1_size].weight;
			}
		}

		for (int b = 0; b < layer2_size; b++)
		{
			for (int a = 0; a < layerc_size; a++)
			{
				sync[a + b * layerc_size].weight = model.sync[a + b * layerc_size].weight;
			}
		}
	}

	long long aa;
	for (aa = 0; aa < direct_size; aa++)
	{
		syn_d[aa] = model.syn_d[aa];
	}

	if (fea_matrix_used)
	{
		if (fea_matrix == NULL) fea_matrix = new real[vocab->vocabSize() * fea_size];
		for (int b = 0; b < vocab->vocabSize(); b++)
		{
			for (int a = 0; a < fea_size; a++)
			{
				fea_matrix[a + b * fea_size] = model.fea_matrix[a + b * fea_size];
			}
		}
	}
}


//Return -1 for oovs
int RNN::getWordIndex(string &str, bool wordIndexed)
{	
	if(wordIndexed) {
		return atoi(str.c_str());
	} else {
		return vocab->getWordId(str);
	}
}


//back up neuron values and synps
//void RNN::saveWeights()      //saves current weights and unit activations
//{
//	int a,b;
//
//	for (a=0; a<layer0_size; a++) {
//		neu0b[a].ac=neu0[a].ac;
//		neu0b[a].er=neu0[a].er;
//	}
//
//	for (a=0; a<layer1_size; a++) {
//		neu1b[a].ac=neu1[a].ac;
//		neu1b[a].er=neu1[a].er;
//	}
//
//	for (a=0; a<layerc_size; a++) {
//		neucb[a].ac=neuc[a].ac;
//		neucb[a].er=neuc[a].er;
//	}
//
//	for (a=0; a<layer2_size; a++) {
//		neu2b[a].ac=neu2[a].ac;
//		neu2b[a].er=neu2[a].er;
//	}
//
//	for (b=0; b<layer1_size; b++) {
//		for (a=0; a<layer0_size; a++) {
//			syn0b[a+b*layer0_size].weight=syn0[a+b*layer0_size].weight;
//		}
//	}
//
//	for (b=0; b<layer1_size; b++) {
//		for (a=0; a<fea_size; a++) {
//			synfb[a+b*fea_size].weight=synf[a+b*fea_size].weight;
//		}
//	}
//
//	for (b=0; b<layer2_size; b++) {
//		for (a=0; a<fea_size; a++) {
//			synfob[a+b*fea_size].weight=synfo[a+b*fea_size].weight;
//		}
//	}
//
//	if (layerc_size>0) {
//		for (b=0; b<layerc_size; b++) {
//			for (a=0; a<layer1_size; a++) {
//				syn1b[a+b*layer1_size].weight=syn1[a+b*layer1_size].weight;
//			}
//		}
//
//		for (b=0; b<layer2_size; b++) {
//			for (a=0; a<layerc_size; a++) {
//				syncb[a+b*layerc_size].weight=sync[a+b*layerc_size].weight;
//			}
//		}
//	}
//	else {
//		for (b=0; b<layer2_size; b++) {
//			for (a=0; a<layer1_size; a++) {
//				syn1b[a+b*layer1_size].weight=syn1[a+b*layer1_size].weight;
//			}
//		}
//	}
//
//	for (a=0; a<direct_size; a++) syn_db[a]=syn_d[a];
//}

//void RNN::restoreWeights()      //restores current weights and unit activations from backup copy
//{
//	int a,b;
//
//	for (a=0; a<layer0_size; a++) {
//		neu0[a].ac=neu0b[a].ac;
//		neu0[a].er=neu0b[a].er;
//	}
//
//	for (a=0; a<layer1_size; a++) {
//		neu1[a].ac=neu1b[a].ac;
//		neu1[a].er=neu1b[a].er;
//	}
//
//	for (a=0; a<layerc_size; a++) {
//		neuc[a].ac=neucb[a].ac;
//		neuc[a].er=neucb[a].er;
//	}
//
//	for (a=0; a<layer2_size; a++) {
//		neu2[a].ac=neu2b[a].ac;
//		neu2[a].er=neu2b[a].er;
//	}
//
//	for (b=0; b<layer1_size; b++) {
//		for (a=0; a<layer0_size; a++) {
//			syn0[a+b*layer0_size].weight=syn0b[a+b*layer0_size].weight;
//		}
//	}
//
//	for (b=0; b<layer1_size; b++) {
//		for (a=0; a<fea_size; a++) {
//			synf[a+b*fea_size].weight=synfb[a+b*fea_size].weight;
//		}
//	}
//
//	for (b=0; b<layer2_size; b++) {
//		for (a=0; a<fea_size; a++) {
//			synfo[a+b*fea_size].weight=synfob[a+b*fea_size].weight;
//		}
//	}
//
//	if (layerc_size>0) {
//		for (b=0; b<layerc_size; b++) {
//			for (a=0; a<layer1_size; a++) {
//				syn1[a+b*layer1_size].weight=syn1b[a+b*layer1_size].weight;
//			}
//		}
//
//		for (b=0; b<layer2_size; b++) {
//			for (a=0; a<layerc_size; a++) {
//				sync[a+b*layerc_size].weight=syncb[a+b*layerc_size].weight;
//			}
//		}
//	}
//	else {
//		for (b=0; b<layer2_size; b++) {
//			for (a=0; a<layer1_size; a++) {
//				syn1[a+b*layer1_size].weight=syn1b[a+b*layer1_size].weight;
//			}
//		}
//	}
//
//	for (a=0; a<direct_size; a++) syn_d[a]=syn_db[a];
//}

//back up hidden layer neurons
//void RNN::saveContext()		//useful for n-best list processing
//{
//	int a;
//	for (a=0; a<layer1_size; a++) {
//		neu1b[a].ac=neu1[a].ac;
//	}
//}
//
////restore hidden layer neurons
//void RNN::restoreContext()
//{
//	int a;
//	for (a=0; a<layer1_size; a++) {
//		neu1[a].ac=neu1b[a].ac;
//	}
//}
//
////back up hidden layer neurons 2
//void RNN::saveContext2()
//{
//	int a;
//	for (a=0; a<layer1_size; a++) {
//		neu1b2[a].ac=neu1[a].ac;
//	}
//}
//
////restore hidden layer neurons 2
//void RNN::restoreContext2()
//{
//	int a;
//	for (a=0; a<layer1_size; a++) {
//		neu1[a].ac=neu1b2[a].ac;
//	}
//}

//init neurons, synapse, and class_words, layer0,
//layer1, and layer2 sizes should be initialized before this call
void RNN::initNet(bool useRandom)
{
	int a, b;
	//if (!fea_matrix_file.empty()) {
	//	fea_matrix_used=1;		//feature matrix file was set
	//}
	//untested code. Matrix features file format
	//word1 topic1Score topic2Score ... topicNScore
	//word2 topic1Score topic2Score ... topicNScore
	//if (fea_matrix_used) {
	//	int topics=0;
	//	FILE *f=fopen(fea_matrix_file.c_str(), "rb");
	//	float fl;
	//	//double max;
	//	char st[1000];

	//	if (f==NULL) {
	//		printf("Feature matrix file not found\n");
	//		exit(1);
	//	}

	//	a=0;
	//	while (1) {
	//		a=fgetc(f);
	//		if (feof(f)) break;
	//		if (a==' ') topics++;
	//		if (a=='\n') break;
	//	}
	//	fclose(f);

	//	if (debug_mode>0) {
	//		printf("Size of feature vectors: %d\n", topics);
	//	}
	//	fea_matrix=(real *)calloc(vocab->vocabSize()*topics, sizeof(real));

	//	fea_size=topics;

	//	for (b=0; b<vocab->vocabSize(); b++) {
	//		fea_matrix[b]=10000;	//means that the prob. dist. is undefined
	//	}
	//	f=fopen(fea_matrix_file.c_str(), "rb");
	//	for (b=0; b<vocab->vocabSize(); b++) {  //can be less
	//		fscanf(f, "%s", st);
	//		c=vocab->getWordId(st);
	//		//if (c<0) c=vocab->vocabSize()-1;
	//		//if (c>=vocab->vocabSize()) c=vocab->vocabSize()-1;

	//		for (a=0; a<fea_size; a++) {
	//			fscanf(f, "%f", &fl);
	//			if (feof(f)) break;
	//			fea_matrix[c+a*vocab->vocabSize()]=fl;
	//		}
	//		if (feof(f)) break;
	//	}
	//	fclose(f);
	//} //end if (fea_matrix_used)

	fea_matrix=NULL;
	neu0=NULL;
	neuf=NULL;
	neu1=NULL;
	neuc=NULL;
	neu2=NULL;

	syn0=NULL;
	synf=NULL;
	synfo=NULL;
	syn1=NULL;
	sync=NULL;
	syn_d=NULL;
	//syn_db=NULL;
	////backup
	//neu0b=NULL;
	//neufb=NULL;
	//neu1b=NULL;
	//neucb=NULL;
	//neu2b=NULL;

	//neu1b2=NULL;

	//syn0b=NULL;
	//synfb=NULL;
	//synfob=NULL;
	//syn1b=NULL;
	//syncb=NULL;

	bptt_history=NULL;
	bptt_hidden=NULL;
	bptt_fea=NULL;
	bptt_syn0=NULL;
	bptt_synf=NULL;

	//layer0_size=vocab->vocabSize()+layer1_size;
	//layer2_size=vocab->vocabSize()+vocab->classSize;

	neu0=(struct neuron *)calloc(layer0_size, sizeof(struct neuron));
	neuf=(struct neuron *)calloc(fea_size, sizeof(struct neuron));
	neu1=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
	neuc=(struct neuron *)calloc(layerc_size, sizeof(struct neuron));
	neu2=(struct neuron *)calloc(layer2_size, sizeof(struct neuron));


	syn0=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
	synf=(struct synapse *)calloc(fea_size*layer1_size, sizeof(struct synapse));
	synfo=(struct synapse *)calloc(fea_size*layer2_size, sizeof(struct synapse));
	if (layerc_size==0) {
		syn1=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
	} else {
		syn1=(struct synapse *)calloc(layer1_size*layerc_size, sizeof(struct synapse));
		sync=(struct synapse *)calloc(layerc_size*layer2_size, sizeof(struct synapse));
	}

	if (syn1==NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	if (layerc_size>0) if (sync==NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}
	syn_d=(direct_t *)calloc((long long)direct_size, sizeof(direct_t));
	if (syn_d==NULL) {
		printf("Memory allocation for direct connections failed (requested %lld bytes)\n", (long long)direct_size * (long long)sizeof(direct_t));
		exit(1);
	}

	//neu0b=(struct neuron *)calloc(layer0_size, sizeof(struct neuron));
	//neufb=(struct neuron *)calloc(fea_size, sizeof(struct neuron));
	//neu1b=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
	//neucb=(struct neuron *)calloc(layerc_size, sizeof(struct neuron));
	//neu1b2=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
	//neu2b=(struct neuron *)calloc(layer2_size, sizeof(struct neuron));

	//syn0b=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
	//synfb=(struct synapse *)calloc(fea_size*layer1_size, sizeof(struct synapse));
	//synfob=(struct synapse *)calloc(fea_size*layer2_size, sizeof(struct synapse));
	////syn1b=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
	//if (layerc_size==0)
	//	syn1b=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
	//else {
	//	syn1b=(struct synapse *)calloc(layer1_size*layerc_size, sizeof(struct synapse));
	//	syncb=(struct synapse *)calloc(layerc_size*layer2_size, sizeof(struct synapse));
	//}

	//if (syn1b==NULL) {
	//	printf("Memory allocation failed\n");
	//	exit(1);
	//}

	//syn_db=(direct_t *)calloc((long long)direct_size, sizeof(direct_t));

	for (a=0; a<layer0_size; a++) {
		neu0[a].ac=0;
		neu0[a].er=0;
	}

	for (a=0; a<fea_size; a++) {
		neuf[a].ac=0;
		neuf[a].er=0;
	}

	for (a=0; a<layer1_size; a++) {
		neu1[a].ac=0;
		neu1[a].er=0;
	}

	for (a=0; a<layerc_size; a++) {
		neuc[a].ac=0;
		neuc[a].er=0;
	}

	for (a=0; a<layer2_size; a++) {
		neu2[a].ac=0;
		neu2[a].er=0;
	}

	for (b=0; b<layer1_size; b++) {
		for (a=0; a<layer0_size; a++) {
			if(useRandom) {
				syn0[a+b*layer0_size].weight=Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1);
			} else {
				syn0[a+b*layer0_size].weight=0;
			}
		}
	}

	for (b=0; b<layer1_size; b++) {
		for (a=0; a<fea_size; a++) {
			if(useRandom) {
				synf[a+b*fea_size].weight=Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1);
			} else {
				synf[a+b*fea_size].weight = 0;
			}
		}
	}

	for (b=0; b<layer2_size; b++) {
		for (a=0; a<fea_size; a++) {
			if(useRandom) {
				synfo[a+b*fea_size].weight=Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1);
			} else {
				synfo[a+b*fea_size].weight=0;
			}
		}
	}

	if (layerc_size>0) {
		for (b=0; b<layerc_size; b++) {
			for (a=0; a<layer1_size; a++) {
				if(useRandom) {
					syn1[a+b*layer1_size].weight=Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1);
				} else {
					syn1[a+b*layer1_size].weight=0;
				}
			}
		}

		for (b=0; b<layer2_size; b++) {
			for (a=0; a<layerc_size; a++) {
				if(useRandom) {
					sync[a+b*layerc_size].weight=Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1);
				} else {
					sync[a+b*layerc_size].weight=0;
				}
			}
		}
	} else {
		for (b=0; b<layer2_size; b++) {
			for (a=0; a<layer1_size; a++) {
				if(useRandom) {
					syn1[a+b*layer1_size].weight=Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1);
				} else {
					syn1[a+b*layer1_size].weight=0;
				}
			}
		}
	}

	long long aa;
	for (aa=0; aa<direct_size; aa++) {
		if(useRandom) {
			syn_d[aa]=Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1)+Utils::random(-0.1, 0.1);
			//syn_db[aa] = syn_d[aa];
		} else {
			syn_d[aa]=0;
			//syn_db[aa]=syn_d[aa];
		}
	}
	if (bptt>0) {
		bptt_history=(int *)calloc((bptt+bptt_block+10), sizeof(int));
		for (a=0; a<bptt+bptt_block; a++) {
			bptt_history[a]=-1;
		}
		//
		bptt_hidden=(neuron *)calloc((bptt+bptt_block+1)*layer1_size, sizeof(neuron));
		for (a=0; a<(bptt+bptt_block)*layer1_size; a++) {
			bptt_hidden[a].ac=0;
			bptt_hidden[a].er=0;
		}
		//
		bptt_fea=(neuron *)calloc((bptt+bptt_block+2)*fea_size, sizeof(neuron));
		for (a=0; a<(bptt+bptt_block)*fea_size; a++) {
			bptt_fea[a].ac=0;
		}
		//
		bptt_syn0=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
		if (bptt_syn0==NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}
		//
		bptt_synf=(struct synapse *)calloc(fea_size*layer1_size, sizeof(struct synapse));
		if (bptt_synf==NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}
	}
}

void RNN::saveNet(string model_file, bool includeVocab)       //will save the whole network structure                                                        
{
	ofstream writer(model_file, ios::out | ios::binary);
	int a, b;
	float fl;

	//save feature file
	//featureIndexer->toString(model_file + "Features");

	writer << "version: " << version << endl;
	writer << "random seed: " << rand_seed << endl;
	string fids;
	Utils::vector2Str(context_ids, ":", fids);
	writer << "context feature ids: " << fids << endl;
	writer << "binary file: " << Utils::bool2Str(file_binary) << endl << endl;
	writer << "training data file: " << train_file << endl;
	writer << "validation data file: " << valid_file << endl << endl;
	writer << "last probability of validation data: " << llogpValid << endl;
	writer << "current probability of validation data: " << logpValid << endl;
	writer << "number of finished iterations: " << iter << endl;
	writer << "current position in training data: " << train_cur_pos << endl;
	writer << "last probability of training data: " << llogpTrain << endl;
	writer << "current probability of training data: " << logpTrain << endl;
	writer << "# of training words: " << train_words << endl;
	writer << "input layer size: " << layer0_size << endl;
	writer << "feature size: " << fea_size << endl;
	writer << "feature matrix used: " << Utils::bool2Str(fea_matrix_used) << endl;
	writer << "feature gamma: " << feature_gamma << endl;
	writer << "hidden layer size: " << layer1_size << endl;
	writer << "compression layer size: " << layerc_size << endl;
	writer << "output layer size: " << layer2_size << endl;
	writer << "direct connections: " << direct_size << endl;
	writer << "direct order: " << direct_order << endl;
	writer << "direct word size: " << direct_word_size << endl;
	writer << "direct class size: " << direct_class_size << endl;
	writer << "bptt: " << bptt << endl;
	writer << "bptt block: " << bptt_block << endl;
	writer << "independent sentences mode: " << Utils::bool2Str(independent) << endl;
	writer << "starting learning rate: " << starting_alpha << endl;
	writer << "current learning rate: " << alpha << endl;
	writer << "learning rate decrease: " << Utils::bool2Str(alpha_divide) << endl;
	writer << "regularization: " << beta << endl;
	writer << "vocab size: " << vocab->vocabSize() << endl;

	if(includeVocab) {
		writer << endl;
		writer << "Vocabulary:" << endl;
		vocab->toString(writer);
	}

	writer.precision(4);

	if (file_binary==false) {
		writer << "\nHidden layer activation:\n";
		for (a=0; a<layer1_size; a++) {
			writer << neu1[a].ac << endl;
		}
	}
	if (file_binary==true) {
		writer << endl; //adding a new line for easy vocab reading
		for (a=0; a<layer1_size; a++) {
			fl=neu1[a].ac;
			writer.write((char*)&fl, sizeof(float));
		}
	}
	//////////
	if (file_binary==false) {
		writer << "\nWeights 0->1:\n";
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<layer0_size; a++) {
				writer <<  syn0[a+b*layer0_size].weight << endl;
			}
		}
	}
	if (file_binary==true) {
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<layer0_size; a++) {
				fl=syn0[a+b*layer0_size].weight;
				writer.write((char*)&fl, sizeof(float));
			}
		}
	}
	//////////
	if (file_binary==false) {
		writer << "\nWeights fea->1:\n";
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<fea_size; a++) {
				writer << synf[a+b*fea_size].weight << endl;
			}
		}
		//
		writer << "\nWeights fea->out:\n";
		for (b=0; b<layer2_size; b++) {
			for (a=0; a<fea_size; a++) {
				writer << synfo[a+b*fea_size].weight <<endl;
			}
		}
	}
	if (file_binary==true) {
		for (b=0; b<layer1_size; b++) {
			for (a=0; a<fea_size; a++) {
				fl=synf[a+b*fea_size].weight;
				writer.write((char*)&fl, sizeof(float));
			}
		}
		//
		for (b=0; b<layer2_size; b++) {
			for (a=0; a<fea_size; a++) {
				fl=synfo[a+b*fea_size].weight;
				writer.write((char*)&fl, sizeof(float));
			}
		}
	}
	/////////
	if (file_binary==false) {
		if (layerc_size>0) {
			writer << "\n\nWeights 1->c:\n";
			for (b=0; b<layerc_size; b++) {
				for (a=0; a<layer1_size; a++) {
					writer <<  syn1[a+b*layer1_size].weight << endl;
				}
			}

			writer << "\n\nWeights c->2:\n";
			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layerc_size; a++) {
					writer << sync[a+b*layerc_size].weight << endl;
				}
			}
		}
		else
		{
			writer << "\n\nWeights 1->2:\n";
			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layer1_size; a++) {
					writer <<  syn1[a+b*layer1_size].weight << endl;
				}
			}
		}
	}
	if (file_binary==true) {
		if (layerc_size>0) {
			for (b=0; b<layerc_size; b++) {
				for (a=0; a<layer1_size; a++) {
					fl=syn1[a+b*layer1_size].weight;
					writer.write((char*)&fl, sizeof(float));
				}
			}

			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layerc_size; a++) {
					fl=sync[a+b*layerc_size].weight;
					writer.write((char*)&fl, sizeof(float));
				}
			}
		}
		else
		{
			for (b=0; b<layer2_size; b++) {
				for (a=0; a<layer1_size; a++) {
					fl=syn1[a+b*layer1_size].weight;
					writer.write((char*)&fl, sizeof(float));
				}
			}
		}
	}
	////////
	if (file_binary==false) {
		writer << "\nDirect connections:\n";
		long long aa;
		for (aa=0; aa<direct_size; aa++) {
			writer << syn_d[aa] << endl;
		}
	}
	if (file_binary==true) {
		long long aa;
		for (aa=0; aa<direct_size; aa++) {
			fl=syn_d[aa];
			writer.write((char*)&fl, sizeof(float));
		}
	}
	////////
	if (fea_matrix_used) {
		if (file_binary==false) {
			writer << "\nFeature matrix:\n";
			for (b=0; b<vocab->vocabSize(); b++) {
				for (a=0; a<fea_size; a++) {
					writer <<  fea_matrix[a+b*fea_size] << endl;
				}
				writer <<  "\n";
			}
		}
		if (file_binary==true) {
			for (b=0; b<vocab->vocabSize(); b++) {
				for (a=0; a<fea_size; a++) {
					fl=fea_matrix[a+b*fea_size];
					writer.write((char*)&fl, sizeof(float));
				}
			}
		}
	}
	writer.close();
}

//cleans all activations and error vectors
void RNN::netFlush()   
{
	int a;

	for (a=0; a<layer0_size-layer1_size; a++) {  // set activations indicating word inputs to 0
		neu0[a].ac=0;
		neu0[a].er=0;
	}

	// set activations of the hidden units
	for (a=layer0_size-layer1_size; a<layer0_size; a++) {   //last hidden layer is initialized to vector of 0.1 values to prevent unstability
		//	neu0[a].ac=1.0;
		neu0[a].ac=0.1;
		neu0[a].er=0;
	}

	for (a=0; a<layer1_size; a++) {
		neu1[a].ac=0;
		neu1[a].er=0;
	}

	for (a=0; a<layerc_size; a++) {
		neuc[a].ac=0;
		neuc[a].er=0;
	}

	for (a=0; a<layer2_size; a++) {
		neu2[a].ac=0;
		neu2[a].er=0;
	}
}

//cleans hidden layer activation + bptt history
void RNN::netReset()   
{
	int a, b;

	for (a=0; a<layer1_size; a++) {  // set hidden unit activations
		neu1[a].ac=1.0;
	}

	copyHiddenLayerToInput();

	if (bptt>0) {
		for (a=1; a<bptt+bptt_block; a++) {
			bptt_history[a]=0;
		}
		for (a=bptt+bptt_block-1; a>1; a--) {
			for (b=0; b<layer1_size; b++) {
				bptt_hidden[a*layer1_size+b].ac=0;
				bptt_hidden[a*layer1_size+b].er=0;
			}
		}
	}

	for (a=0; a<MAX_NGRAM_ORDER; a++) {
		history[a]=0;
	}
}

void RNN::matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type)
{
	int a, b;
	real val1, val2, val3, val4;
	real val5, val6, val7, val8;

	if (type==0) {		//ac mod
		//#pragma simd statement

		for (b=0; b<(to-from)/8; b++) {
			val1=0;
			val2=0;
			val3=0;
			val4=0;

			val5=0;
			val6=0;
			val7=0;
			val8=0;
			//#pragma omp parallel for
			for (a=from2; a<to2; a++) {
				val1 += srcvec[a].ac * srcmatrix[a+(b*8+from+0)*matrix_width].weight;
				val2 += srcvec[a].ac * srcmatrix[a+(b*8+from+1)*matrix_width].weight;
				val3 += srcvec[a].ac * srcmatrix[a+(b*8+from+2)*matrix_width].weight;
				val4 += srcvec[a].ac * srcmatrix[a+(b*8+from+3)*matrix_width].weight;

				val5 += srcvec[a].ac * srcmatrix[a+(b*8+from+4)*matrix_width].weight;
				val6 += srcvec[a].ac * srcmatrix[a+(b*8+from+5)*matrix_width].weight;
				val7 += srcvec[a].ac * srcmatrix[a+(b*8+from+6)*matrix_width].weight;
				val8 += srcvec[a].ac * srcmatrix[a+(b*8+from+7)*matrix_width].weight;
			}
			dest[b*8+from+0].ac += val1;
			dest[b*8+from+1].ac += val2;
			dest[b*8+from+2].ac += val3;
			dest[b*8+from+3].ac += val4;

			dest[b*8+from+4].ac += val5;
			dest[b*8+from+5].ac += val6;
			dest[b*8+from+6].ac += val7;
			dest[b*8+from+7].ac += val8;
		}

		for (b=b*8; b<to-from; b++) {
			for (a=from2; a<to2; a++) {
				dest[b+from].ac += srcvec[a].ac * srcmatrix[a+(b+from)*matrix_width].weight;
			}
		}
	}
	else {		//er mod
		//#pragma simd statement

		for (a=0; a<(to2-from2)/8; a++) {
			val1=0;
			val2=0;
			val3=0;
			val4=0;

			val5=0;
			val6=0;
			val7=0;
			val8=0;
			//#pragma omp parallel for
			for (b=from; b<to; b++) {
				val1 += srcvec[b].er * srcmatrix[a*8+from2+0+b*matrix_width].weight;
				val2 += srcvec[b].er * srcmatrix[a*8+from2+1+b*matrix_width].weight;
				val3 += srcvec[b].er * srcmatrix[a*8+from2+2+b*matrix_width].weight;
				val4 += srcvec[b].er * srcmatrix[a*8+from2+3+b*matrix_width].weight;

				val5 += srcvec[b].er * srcmatrix[a*8+from2+4+b*matrix_width].weight;
				val6 += srcvec[b].er * srcmatrix[a*8+from2+5+b*matrix_width].weight;
				val7 += srcvec[b].er * srcmatrix[a*8+from2+6+b*matrix_width].weight;
				val8 += srcvec[b].er * srcmatrix[a*8+from2+7+b*matrix_width].weight;
			}
			dest[a*8+from2+0].er += val1;
			dest[a*8+from2+1].er += val2;
			dest[a*8+from2+2].er += val3;
			dest[a*8+from2+3].er += val4;

			dest[a*8+from2+4].er += val5;
			dest[a*8+from2+5].er += val6;
			dest[a*8+from2+6].er += val7;
			dest[a*8+from2+7].er += val8;
		}

		for (a=a*8; a<to2-from2; a++) {
			for (b=from; b<to; b++) {
				dest[a+from2].er += srcvec[b].er * srcmatrix[a+from2+b*matrix_width].weight;
			}
		}

		if (gradient_cutoff>0) {
			for (a=from2; a<to2; a++) {
				if (dest[a].er>gradient_cutoff) dest[a].er=gradient_cutoff;
				if (dest[a].er<-gradient_cutoff) dest[a].er=-gradient_cutoff;
			}
		}
	}

	//this is normal implementation (about 3x slower):
	/*
	if (type==0) {		//ac mod
	for (b=from; b<to; b++) {
	for (a=from2; a<to2; a++) {
	dest[b].ac += srcvec[a].ac * srcmatrix[a+b*matrix_width].weight;
	}
	}
	}
	else 		//er mod
	if (type==1) {
	for (a=from2; a<to2; a++) {
	for (b=from; b<to; b++) {
	dest[a].er += srcvec[b].er * srcmatrix[a+b*matrix_width].weight;
	}
	}
	}
	*/
}

//experimental
//void RNN::matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type)
//{
//	int a, b;
//	if (type==0) {		//ac mod
//		//#pragma omp parallel for
//		for (b=from; b<to; b++) {
//			for (a=from2; a<to2; a++) {
//				dest[b].ac += srcvec[a].ac * srcmatrix[a+b*matrix_width].weight;
//			}
//		}
//	}
//	else {		//er mod
//		if (type==1) {
//			//#pragma omp parallel for
//			for (a=from2; a<to2; a++) {
//				for (b=from; b<to; b++) {
//					dest[a].er += srcvec[b].er * srcmatrix[a+b*matrix_width].weight;
//				}
//			}
//		}
//	}
//}

void RNN::computeNet(int last_word, int word, bool generate)
{
	int a, b;
	real val;
	double sum;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here

	if (word==-1) return;

	if (last_word!=-1) {
		neu0[last_word].ac=1;
	}
	//erase activations
	for (a=0; a<layer1_size; a++) {
		neu1[a].ac=0;
	}
	for (a=0; a<layerc_size; a++) {
		neuc[a].ac=0;
	}

	//step 0: hidden(t-1) -> hidden(t)
	matrixXvector(neu1, neu0, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 0);

	//step 1: word(t) -> hidden(t)
	for (b=0; b<layer1_size; b++) {
		a=last_word;
		if (a!=-1) {
			neu1[b].ac += neu0[a].ac * syn0[a+b*layer0_size].weight;
		}
	}

	//fea(t) -> hidden(t)
	if (fea_size>0)	{
		matrixXvector(neu1, neuf, synf, fea_size, 0, layer1_size, 0, fea_size, 0);
	}

	//step 2: activate 1      --sigmoid
	for (a=0; a<layer1_size; a++) {
		if (neu1[a].ac>50) neu1[a].ac=50;  //for numerical stability
		if (neu1[a].ac<-50) neu1[a].ac=-50;  //for numerical stability
		val=-neu1[a].ac;
		neu1[a].ac=1/(1+exp(val));
	}

	if (layerc_size>0) {
		matrixXvector(neuc, neu1, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 0);
		//activate compression      --sigmoid
		for (a=0; a<layerc_size; a++) {
			if (neuc[a].ac>50) neuc[a].ac=50;  //for numerical stability
			if (neuc[a].ac<-50) neuc[a].ac=-50;  //for numerical stability
			val=-neuc[a].ac;
			neuc[a].ac=1/(1+exp(val));
		}
	}

	//super classes
	if(vocab->superClassSize > 1) {
		//step 3': 1->2 super class
		for (b=vocab->vocabSize() + vocab->classSize; b<layer2_size; b++) {
			neu2[b].ac=0;
		}
		if (layerc_size>0) {
			matrixXvector(neu2, neuc, sync, layerc_size, vocab->vocabSize()+ vocab->classSize, layer2_size, 0, layerc_size, 0);
		}
		else
		{
			matrixXvector(neu2, neu1, syn1, layer1_size, vocab->vocabSize()+ vocab->classSize, layer2_size, 0, layer1_size, 0);
		}

		//fea to out class, zhiheng, add later
		/*if (fea_size>0)	{
		matrixXvector(neu2, neuf, synfo, fea_size, vocab->vocabSize(), layer2_size, 0, fea_size, 0);
		}*/

		//step 4': apply direct connections to super classes
		if (direct_size>0) {
			//ngram order (0, 1, 2 etc) to its feature index
			vector<unsigned long long> hash;	//this will hold pointers to syn_d that contains hash parameters
			getMEFeaIds(hash, 0, -1);

			for (a=vocab->vocabSize()+ vocab->classSize; a<layer2_size; a++) {
				for (b=0; b<hash.size(); b++) {
					neu2[a].ac+=syn_d[hash[b]];		//apply current parameter and move to the next one	
					hash[b]++;
				}
			}		
		}

		//step 5': softmax on super classes
		sum=0;
		for (a=vocab->vocabSize()+ vocab->classSize; a<layer2_size; a++) {
			if (neu2[a].ac>50) neu2[a].ac=50;  //for numerical stability
			if (neu2[a].ac<-50) neu2[a].ac=-50;  //for numerical stability
			val=exp(neu2[a].ac);
			sum+=val;
			neu2[a].ac=val;
		}
		for (a=vocab->vocabSize()+ vocab->classSize; a<layer2_size; a++) {
			neu2[a].ac/=sum;         //output layer activations now sum exactly to 1
		}
	}

	//step 3: 1->2 class
	int superClassIndex = -1;
	int start = vocab->vocabSize();
	int end = start + vocab->classSize;
	if(vocab->superClassSize > 1) {
		superClassIndex = vocab->getSuperClass(word);
		start = vocab->vocabSize() + vocab->getClassIdFromSuperClass(superClassIndex, 0);
		end = start + vocab->numClassesInSameSuperClass(word);
	}
	for (b=start; b<end; b++) {
		neu2[b].ac=0;
	}
	if (layerc_size>0) {
		matrixXvector(neu2, neuc, sync, layerc_size, start, end, 0, layerc_size, 0);
	}
	else
	{
		matrixXvector(neu2, neu1, syn1, layer1_size, start, end, 0, layer1_size, 0);
	}

	//fea to out class
	if (fea_size>0)	{
		matrixXvector(neu2, neuf, synfo, fea_size, start, end, 0, fea_size, 0);
	}

	//step 4: apply direct connections to classes
	if (direct_size>0) {
		//ngram order (0, 1, 2 etc) to its feature index
		vector<unsigned long long> hash;	//this will hold pointers to syn_d that contains hash parameters
		getMEFeaIds(hash, 1, superClassIndex);

		for (a=start; a<end; a++) {
			for (b=0; b<hash.size(); b++) {
				neu2[a].ac+=syn_d[hash[b]];		//apply current parameter and move to the next one	
				hash[b]++;
			}
		}		
	}

	//sparse feature index
	//vector<int> labels;
	//vector<unsigned long> featureIds;
	//featureIndexer->getFeatures(history, direct_order, labels, featureIds);
	//for (b=0; b<labels.size(); b++) {
	//	neu2[labels[b]].ac+=featureIndexer->fvs[featureIds[b]];		//apply current parameter and move to the next one	
	//	//hash[b]++;
	//}

	//step 5: activation 2   --softmax on classes
	sum=0;
	for (a=start; a<end; a++) {
		if (neu2[a].ac>50) neu2[a].ac=50;  //for numerical stability
		if (neu2[a].ac<-50) neu2[a].ac=-50;  //for numerical stability
		val=exp(neu2[a].ac);
		sum+=val;
		neu2[a].ac=val;
	}

	for (a=start; a<end; a++) {
		neu2[a].ac/=sum;         //output layer activations now sum exactly to 1
	}
	if (generate == true) {
		return;	//if we generate words, we don't know what current word is -> only classes are estimated and word is selected in testGen()
	}

	//step 6: 1->2 word
	start = vocab->getWordIdFromClass(vocab->getClass(word), 0);
	end = start + vocab->numWordsInSameClass(word);
	int classIndex = vocab->getClass(word);
	if (word!=-1) {
		for (int i=start; i<end; i++) {
			neu2[i].ac=0;
		}
		if (layerc_size>0) {
			matrixXvector(neu2, neuc, sync, layerc_size, start, end, 0, layerc_size, 0);
		}
		else
		{
			//the words in vocab (and in output layer) are sorted in ascending order of their classes
			matrixXvector(neu2, neu1, syn1, layer1_size, start, end, 0, layer1_size, 0);
		}
	}

	//fea to out word
	if (fea_size>0)	{
		matrixXvector(neu2, neuf, synfo, fea_size, 
			start,end, 0, fea_size, 0);
	}

	//step 7: apply direct connections to words
	if (word!=-1) {		
		if (direct_size>0) {
			//ngram order (0, 1, 2 etc) to its feature index
			vector<unsigned long long> hash;	//this will hold pointers to syn_d that contains hash parameters
			getMEFeaIds(hash, 2, classIndex);

			for (a = start; a < end; a++) {
				for (b=0; b<hash.size(); b++) {					
					neu2[a].ac+=syn_d[hash[b]];		
					hash[b]++;
					hash[b]=hash[b]%direct_size;
				}
			}
		}
	}

	//step 8: activation 2   --softmax on words
	sum=0;
	if (word!=-1) {
		for (a = start; a < end; a++)
		{
			if (neu2[a].ac>50) neu2[a].ac=50;  //for numerical stability
			if (neu2[a].ac<-50) neu2[a].ac=-50;  //for numerical stability
			val=exp(neu2[a].ac);
			sum+=val;
			neu2[a].ac=val;
		}
		for (a = start; a < end; a++)
		{
			neu2[a].ac /= sum;
		}
	}
}

//get maxent feature index via a hash function. type=0:superClass,
//type=1:class, type=2:word. parentIndex is the index of type's parent. For
//example, if type=1, parentIndex is the index of superClass.
void RNN::getMEFeaIds(vector<unsigned long long> &hash, int type, int parentIndex) {	
	//first attempt to add features ids for super classes

	hash.clear();
	int a, b;
	unsigned long long featureId;
	for (a=0; a<direct_order; a++) {
		b=0;
		if (a>0) {
			if (history[a-1]==-1) {
				break;	//if OOV was in history, do not use this N-gram feature and higher orders
			}
		}
		if(type == 0 || type == 1 || type == 2) { 
			//maxent like features includes current word's class, but not past words' classes
			//featureId=PRIMES[type + 0]*PRIMES[type + 1]*(unsigned long long)(parentIndex+2); 
			featureId=PRIMES[type + 5]*PRIMES[type + 6]*(unsigned long long)(parentIndex+5);
		} else {
			cerr << "maxent feature type " << type << " not supported" << endl;
			exit(1);
		}

		for (b=1; b<=a; b++) {
			featureId+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);	//update hash value based on words from the history
		}
		if(type == 0 || type == 1) {
			featureId=featureId%(direct_size/2);		//make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
		} else {
			featureId=(featureId%(direct_size/2))+(direct_size)/2;
		}
		hash.push_back(featureId);
	} 

	//second attempt to get backward compatibility to previous versions, large pplx difference between
	//base and HC, to investigate
	/*
	hash.clear();
	int a, b;
	unsigned long long featureId;
	for (a=0; a<direct_order; a++) {
	b=0;
	if (a>0) {
	if (history[a-1]==-1) {
	break;	//if OOV was in history, do not use this N-gram feature and higher orders
	}
	}
	if( type == 0) { //super class
	featureId=PRIMES[2]*PRIMES[5];
	} else if(type == 1) { //class
	if(parentIndex == -1) { //backward compatible
	featureId=PRIMES[0]*PRIMES[1];
	} else {			
	featureId *= PRIMES[4]*PRIMES[8]*(unsigned long long)(parentIndex+1);
	}
	} else if( type == 2) {
	//maxent like features includes current word's class, but not past words' classes
	featureId=PRIMES[0]*PRIMES[1]*(unsigned long long)(parentIndex+1); 
	}
	for (b=1; b<=a; b++) {
	featureId+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);	//update hash value based on words from the history
	}
	if(type == 0 || type == 1) {
	featureId=featureId%(direct_size/2);		//make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
	} else {
	featureId=(featureId%(direct_size/2))+(direct_size)/2;
	}
	hash.push_back(featureId);
	}
	*/

	/* to add later
	//add individual words features
	for (a = 1; a<direct_word_size;a++) {		
	if(forClass) {
	featureId = PRIMES[3]*PRIMES[4]*(unsigned long long)(history[a-1]+1);
	featureId=featureId%(direct_size/2);		//make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
	} else {
	featureId = PRIMES[3]*PRIMES[4]*(unsigned long long)(history[a-1]+1)*(unsigned long long)(classIndex+1);
	featureId=(featureId%(direct_size/2))+(direct_size)/2;
	}
	hash.push_back(featureId);
	}

	//add individual classes features
	for (a = 1; a<direct_class_size;a++) {		
	if(forClass) {
	featureId = PRIMES[5]*PRIMES[6]*(unsigned long long)(vocab->getClass(history[a-1])+1);
	featureId=featureId%(direct_size/2);		//make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
	} else {
	featureId = PRIMES[5]*PRIMES[6]*(unsigned long long)(vocab->getClass(history[a-1])+1)*(unsigned long long)(classIndex+1);
	featureId=(featureId%(direct_size/2))+(direct_size)/2;
	}
	hash.push_back(featureId);
	}

	//add context features
	stringstream ss;
	for(int i = 0; i < context_ids.size(); i++) {
	int a = context_ids.at(i);	
	ss.str("");
	ss << a << "-" << event[a];
	string context = ss.str();
	locale loc;                 // the "C" locale
	const collate<char>& coll = use_facet<collate<char> >(loc);
	long h = coll.hash(context.data(),context.data()+context.length());

	if(forClass) {
	featureId = PRIMES[7]*PRIMES[8]*(unsigned long long)(h+1);
	featureId=featureId%(direct_size/2);		//make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
	} else {
	featureId = PRIMES[7]*PRIMES[8]*(unsigned long long)(h+1)*(unsigned long long)(classIndex+1);
	featureId=(featureId%(direct_size/2))+(direct_size)/2;
	}
	hash.push_back(featureId);
	}
	*/
	//temp code
	/*cerr << "\nhistory:";
	for(int i = 0; i < 5; i++) {
	cerr << vocab->getWordStr(history[i]) << " ";
	}
	cerr << endl;
	cerr << "class:" << forClass << " classIndex:" << classIndex << endl;
	cerr << "feature hashes:";
	for(int i = 0; i < hash.size(); i++) {
	cerr << hash[i] << " ";
	}
	int k = 0;*/
}

//learn synapse from word pair
void RNN::learnNet(int last_word, int word)
{
	int a, b, c, t, step;
	real beta2, beta3;

	beta2=beta*alpha;
	beta3=beta2*1;	//beta3 can be possibly larger than beta2, as that is useful on small datasets (if the final model is to be interpolated wich backoff model) - todo in the future

	if (word==-1) return;

	int classIndex = vocab->getClass(word);	
	int start = vocab->getWordIdFromClass(classIndex, 0);
	int end = start + vocab->numWordsInSameClass(word);

	int superClassIndex = -1;
	int startClass = vocab->vocabSize();
	int endClass = startClass + vocab->classSize;
	if(vocab->superClassSize > 1) {
		superClassIndex = vocab->getSuperClass(word);
		startClass = vocab->vocabSize() + vocab->getClassIdFromSuperClass(superClassIndex, 0);
		endClass = startClass + vocab->numClassesInSameSuperClass(word);	
	}

	//step 8, 5 and 5': compute error vectors
	for (a = start; a < end; a++)
	{
		neu2[a].er=(0-neu2[a].ac);
	}
	neu2[word].er=(1-neu2[word].ac);	//word part	

	for (a=startClass; a<endClass; a++) {
		neu2[a].er=(0-neu2[a].ac);
	}
	neu2[vocab->vocabSize() + classIndex].er=(1-neu2[vocab->vocabSize() + classIndex].ac);	//class part

	if(vocab->superClassSize > 1) {
		for (a=vocab->vocabSize() + vocab->classSize; a<layer2_size; a++) {
			neu2[a].er=(0-neu2[a].ac);
		}
		neu2[vocab->vocabSize() + vocab->classSize + superClassIndex].er=(1-neu2[vocab->vocabSize() + vocab->classSize + superClassIndex].ac);	//super class part
	}
	//flush error
	for (a=0; a<layer1_size; a++) neu1[a].er=0;
	for (a=0; a<layerc_size; a++) neuc[a].er=0;

	//sparse feature index
	//vector<int> labels;
	//vector<unsigned long> featureIds;
	//featureIndexer->getFeatures(history, direct_order, labels, featureIds);
	//for (b=0; b<labels.size(); b++) {
	//	featureIndexer->fvs[featureIds[b]]+=alpha*neu2[labels[b]].er - featureIndexer->fvs[featureIds[b]]*beta3;
	//}

	//step 7
	if (direct_size>0) { //learn direct connections between words
		if (word!=-1) {
			//ngram order (0, 1, 2 etc) to its feature index
			vector<unsigned long long> hash;	//this will hold pointers to syn_d that contains hash parameters
			getMEFeaIds(hash, 2, classIndex);

			for (a = start; a < end; a++)
			{
				for (b=0; b<hash.size(); b++) {					
					syn_d[hash[b]]+=alpha*neu2[a].er - syn_d[hash[b]]*beta3;
					hash[b]++;
					hash[b]=hash[b]%direct_size;
				}
			}
		}
	}

	//step 4: learn direct connections to classes
	if (direct_size>0) {
		//ngram order (0, 1, 2 etc) to its feature index
		vector<unsigned long long> hash;	//this will hold pointers to syn_d that contains hash parameters
		getMEFeaIds(hash, 1, superClassIndex);

		for (a=startClass; a<endClass; a++) {
			for (b=0; b<hash.size(); b++) {
				syn_d[hash[b]]+=alpha*neu2[a].er - syn_d[hash[b]]*beta3;
				hash[b]++;
			}
		}
	}

	//step 4': learn direct connections to super classes
	if (direct_size>0 && vocab->superClassSize > 1) {
		//ngram order (0, 1, 2 etc) to its feature index
		vector<unsigned long long> hash;	//this will hold pointers to syn_d that contains hash parameters
		getMEFeaIds(hash, 0, -1);

		for (a=vocab->vocabSize() + vocab->classSize; a<layer2_size; a++) {
			for (b=0; b<hash.size(); b++) {
				syn_d[hash[b]]+=alpha*neu2[a].er - syn_d[hash[b]]*beta3;
				hash[b]++;
			}
		}
	}


	if (layerc_size>0) {
		matrixXvector(neuc, neu2, sync, layerc_size, start, end, 0, layerc_size, 1);

		t=start*layerc_size;
		for (b = start; b < end; b++)
		{
			if ((counter%10)==0)	//regularization is done every 10. step
				for (a=0; a<layerc_size; a++) sync[a+t].weight+=alpha*neu2[b].er*neuc[a].ac - sync[a+t].weight*beta2;
			else
				for (a=0; a<layerc_size; a++) sync[a+t].weight+=alpha*neu2[b].er*neuc[a].ac;
			t+=layerc_size;
		}
		//
		matrixXvector(neuc, neu2, sync, layerc_size, vocab->vocabSize(), layer2_size, 0, layerc_size, 1);		//propagates errors 2->c for classes

		c=vocab->vocabSize()*layerc_size;
		for (b=vocab->vocabSize(); b<layer2_size; b++) {
			if ((counter%10)==0) {	//regularization is done every 10. step
				for (a=0; a<layerc_size; a++) sync[a+c].weight+=alpha*neu2[b].er*neuc[a].ac - sync[a+c].weight*beta2;	//weight c->2 update
			}
			else {
				for (a=0; a<layerc_size; a++) sync[a+c].weight+=alpha*neu2[b].er*neuc[a].ac;	//weight c->2 update
			}
			c+=layerc_size;
		}

		for (a=0; a<layerc_size; a++) neuc[a].er=neuc[a].er*neuc[a].ac*(1-neuc[a].ac);    //error derivation at compression layer

		////

		matrixXvector(neu1, neuc, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 1);		//propagates errors c->1

		for (b=0; b<layerc_size; b++) {
			for (a=0; a<layer1_size; a++) syn1[a+b*layer1_size].weight+=alpha*neuc[b].er*neu1[a].ac;	//weight 1->c update
		}
	}
	else
	{
		//step 6: error 2->1 for words 
		matrixXvector(neu1, neu2, syn1, layer1_size, start, end, 0, layer1_size, 1);	

		t=start*layer1_size;
		for (b = start; b < end; b++)
		{
			if ((counter%10)==0) {	//regularization is done every 10. step
				for (a=0; a<layer1_size; a++) {
					syn1[a+t].weight+=alpha*neu2[b].er*neu1[a].ac - syn1[a+t].weight*beta2;
				}
			} else {
				//				for (a=0; a<layer1_size; a++) syn1[a+t].weight+=alpha*neu2[b].er*neu1[a].ac;
				const double er = alpha*neu2[b].er;
				for (a=0; a+3<layer1_size; a+=4) {
					syn1[a+t].weight+=er*neu1[a].ac;
					syn1[a+t+1].weight+=er*neu1[a+1].ac;
					syn1[a+t+2].weight+=er*neu1[a+2].ac;
					syn1[a+t+3].weight+=er*neu1[a+3].ac;
				}
				for (; a<layer1_size; a++) {
					syn1[a+t].weight+=er*neu1[a].ac;
				}
			}
			t+=layer1_size;
		}

		//step 3: errors 2->1 for classes
		matrixXvector(neu1, neu2, syn1, layer1_size, startClass, endClass, 0, layer1_size, 1);

		c=startClass*layer1_size;
		for (b=startClass; b<endClass; b++) {
			if ((counter%10)==0) {	//regularization is done every 10. step
				for (a=0; a<layer1_size; a++) {
					syn1[a+c].weight+=alpha*neu2[b].er*neu1[a].ac - syn1[a+c].weight*beta2;	//weight 1->2 update
				}
			}
			else {
				//				for (a=0; a<layer1_size; a++) syn1[a+c].weight+=alpha*neu2[b].er*neu1[a].ac;	//weight 1->2 update
				double er = alpha*neu2[b].er;		
				for (a=0; a+3<layer1_size; a+=4) {
					syn1[a+c].weight+=er*neu1[a].ac;
					syn1[a+c+1].weight+=er*neu1[a+1].ac;
					syn1[a+c+2].weight+=er*neu1[a+2].ac;
					syn1[a+c+3].weight+=er*neu1[a+3].ac;
				}
				for (; a<layer1_size; a++) {
					syn1[a+c].weight+=er*neu1[a].ac;
				}
			}
			c+=layer1_size;
		}

		//step 3': errors 2->1 for super classes
		if(vocab->superClassSize > 1) {
			matrixXvector(neu1, neu2, syn1, layer1_size, vocab->vocabSize()+vocab->classSize, layer2_size, 0, layer1_size, 1);

			c=(vocab->vocabSize()+vocab->classSize)*layer1_size;
			for (b=vocab->vocabSize()+vocab->classSize; b<layer2_size; b++) {
				if ((counter%10)==0) {	//regularization is done every 10. step
					for (a=0; a<layer1_size; a++) {
						syn1[a+c].weight+=alpha*neu2[b].er*neu1[a].ac - syn1[a+c].weight*beta2;	//weight 1->2 update
					}
				}
				else {
					//				for (a=0; a<layer1_size; a++) syn1[a+c].weight+=alpha*neu2[b].er*neu1[a].ac;	//weight 1->2 update
					double er = alpha*neu2[b].er;		
					for (a=0; a+3<layer1_size; a+=4) {
						syn1[a+c].weight+=er*neu1[a].ac;
						syn1[a+c+1].weight+=er*neu1[a+1].ac;
						syn1[a+c+2].weight+=er*neu1[a+2].ac;
						syn1[a+c+3].weight+=er*neu1[a+3].ac;
					}
					for (; a<layer1_size; a++) {
						syn1[a+c].weight+=er*neu1[a].ac;
					}
				}
				c+=layer1_size;
			}
		}
	}

	//direct fea size weights update
	t=start*fea_size;
	for (b = start; b < end; b++)
	{
		for (a=0; a<fea_size; a++) {
			synfo[a+t].weight+=alpha*neu2[b].er*neuf[a].ac;
		}
		t+=fea_size;
	}
	//
	c=vocab->vocabSize()*fea_size;
	for (b=vocab->vocabSize(); b<layer2_size; b++) {
		for (a=0; a<fea_size; a++) {
			synfo[a+c].weight+=alpha*neu2[b].er*neuf[a].ac;	//weight fea->2 update for classes
		}
		c+=fea_size;
	}

	//

	///////////////

	if (bptt<=1) {		//bptt==1 -> normal BP
		for (a=0; a<layer1_size; a++) {
			neu1[a].er=neu1[a].er*neu1[a].ac*(1-neu1[a].ac);    //error derivation at layer 1
		}
		//weight update hidden(t) -> input(t)
		a=last_word;
		if (a!=-1) {
			if ((counter%10)==0)
				for (b=0; b<layer1_size; b++) {
					syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac - syn0[a+b*layer0_size].weight*beta2;
				}
			else
				for (b=0; b<layer1_size; b++) syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac;
		}

		//weight update hidden(t) -> hidden(t-1)
		if ((counter%10)==0) {
			for (b=0; b<layer1_size; b++) for (a=layer0_size-layer1_size; a<layer0_size; a++) syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac - syn0[a+b*layer0_size].weight*beta2;
		}
		else {
			for (b=0; b<layer1_size; b++) for (a=layer0_size-layer1_size; a<layer0_size; a++) syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac;
		}

		//weight update hidden(t) -> fea(t)
		if ((counter%10)==0) {
			for (b=0; b<layer1_size; b++) for (a=0; a<fea_size; a++) synf[a+b*fea_size].weight+=alpha*neu1[b].er*neuf[a].ac - synf[a+b*fea_size].weight*beta2;
		}
		else {
			for (b=0; b<layer1_size; b++) for (a=0; a<fea_size; a++) synf[a+b*fea_size].weight+=alpha*neu1[b].er*neuf[a].ac;
		}
	}
	else		//BPTT
	{
		for (b=0; b<layer1_size; b++) bptt_hidden[b].ac=neu1[b].ac;
		for (b=0; b<layer1_size; b++) bptt_hidden[b].er=neu1[b].er;
		for (b=0; b<fea_size; b++) bptt_fea[b].ac=neuf[b].ac;

		if (((counter%bptt_block)==0) || (independent && (word==0))) {
			for (step=0; step<bptt+bptt_block-2; step++) { // 0 is most recent step
				for (a=0; a<layer1_size; a++) {
					neu1[a].er=neu1[a].er*neu1[a].ac*(1-neu1[a].ac);    //error derivation at layer 1
				}
				//weight update fea->0
				for (b=0; b<layer1_size; b++) 
					for (a=0; a<fea_size; a++) {
						bptt_synf[a+b*fea_size].weight+=alpha*neu1[b].er*bptt_fea[a+step*fea_size].ac;
					}

					//weight update 1->0
					a=bptt_history[step];
					if (a!=-1)
						for (b=0; b<layer1_size; b++) { //update bptt_syn0 rather than syn0
							bptt_syn0[a+b*layer0_size].weight+=alpha*neu1[b].er;//*neu0[a].ac; --should be always set to 1
						}

						for (a=layer0_size-layer1_size; a<layer0_size; a++) 
							neu0[a].er=0;

						//get neu0.err for hidden part from syn0 and neu1.err, backward
						matrixXvector(neu0, neu1, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 1);		//propagates errors 1->0

						//update bptt_syn0 for hidden from neu1.er and neu0.ac
						for (b=0; b<layer1_size; b++) {
							double er = alpha*neu1[b].er;
							int offset = b*layer0_size;
							//						for (a=layer0_size-layer1_size; a<layer0_size; a++) bptt_syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac;
							for (a=layer0_size-layer1_size; a+3<layer0_size; a+=4) {
								bptt_syn0[a+offset].weight+=er*neu0[a].ac;
								bptt_syn0[a+offset+1].weight+=er*neu0[a+1].ac;
								bptt_syn0[a+offset+2].weight+=er*neu0[a+2].ac;
								bptt_syn0[a+offset+3].weight+=er*neu0[a+3].ac;
							}
							for (; a<layer0_size; a++) bptt_syn0[a+offset].weight+=er*neu0[a].ac;
						}

						//update neu1.er from hidden neu0.er and bptt, forward
						for (a=0; a<layer1_size; a++) {		//propagate error from time T-n to T-n-1
							neu1[a].er=neu0[a+layer0_size-layer1_size].er + bptt_hidden[(step+1)*layer1_size+a].er;
						}

						//update neu1.ac and neu0.ac from 
						if (step<bptt+bptt_block-3)
							for (a=0; a<layer1_size; a++) {
								neu1[a].ac=bptt_hidden[(step+1)*layer1_size+a].ac;
								neu0[a+layer0_size-layer1_size].ac=bptt_hidden[(step+2)*layer1_size+a].ac;
							}
			} //end for (step=0; step<bptt+bptt_block-2; step++) {

			for (a=0; a<(bptt+bptt_block)*layer1_size; a++) {
				bptt_hidden[a].er=0;
			}

			for (b=0; b<layer1_size; b++) neu1[b].ac=bptt_hidden[b].ac;		//restore hidden layer after bptt

			//
			for (b=0; b<layer1_size; b++) {		//copy temporary syn0
				if ((counter%10)==0) {
					for (a=layer0_size-layer1_size; a<layer0_size; a++) {
						syn0[a+b*layer0_size].weight+=bptt_syn0[a+b*layer0_size].weight - syn0[a+b*layer0_size].weight*beta2;
						bptt_syn0[a+b*layer0_size].weight=0;
					}
				}
				else {
					for (a=layer0_size-layer1_size; a<layer0_size; a++) {
						syn0[a+b*layer0_size].weight+=bptt_syn0[a+b*layer0_size].weight;
						bptt_syn0[a+b*layer0_size].weight=0;
					}
				}
				//
				if ((counter%10)==0) {
					for (a=0; a<fea_size; a++) {
						synf[a+b*fea_size].weight+=bptt_synf[a+b*fea_size].weight - synf[a+b*fea_size].weight*beta2;
						bptt_synf[a+b*fea_size].weight=0;
					}
				}
				else {
					for (a=0; a<fea_size; a++) {
						synf[a+b*fea_size].weight+=bptt_synf[a+b*fea_size].weight;
						bptt_synf[a+b*fea_size].weight=0;
					}
				}
				//
				if ((counter%10)==0) {
					for (step=0; step<bptt+bptt_block-2; step++) if (bptt_history[step]!=-1) {
						syn0[bptt_history[step]+b*layer0_size].weight+=bptt_syn0[bptt_history[step]+b*layer0_size].weight - syn0[bptt_history[step]+b*layer0_size].weight*beta2;
						bptt_syn0[bptt_history[step]+b*layer0_size].weight=0;
					}
				}
				else {
					for (step=0; step<bptt+bptt_block-2; step++) if (bptt_history[step]!=-1) {
						syn0[bptt_history[step]+b*layer0_size].weight+=bptt_syn0[bptt_history[step]+b*layer0_size].weight;
						bptt_syn0[bptt_history[step]+b*layer0_size].weight=0;
					}
				}
			}
		} //end if (((counter%bptt_block)==0) || (independent && (word==0))) {
	}	//BPTT
}

void RNN::copyHiddenLayerToInput()
{
	int a;
	for (a=0; a<layer1_size; a++) {
		neu0[a+layer0_size-layer1_size].ac=neu1[a].ac;
	}
}

//for feature matrix
void RNN::updateFeatureVector(int w)
{
	double gamma=feature_gamma;
	int a;

	if (w<0) return;

	if (fea_matrix[w]>=1000) return;	//this means that the features for this word were not defined

	if (independent) if (w==0) {	//this will reset the feature vector at the beginning of each sentence
		for (a=0; a<fea_size; a++) neuf[a].ac=0;
	}

	for (a=0; a<fea_size; a++) neuf[a].ac=neuf[a].ac*gamma+fea_matrix[a*vocab->vocabSize()+w]*(1-gamma);
}


void RNN::trainNet(bool wordIndexed, string train_log_file)
{
	int a, b, word, last_word;
	//char log_name[200];
	//FILE *fi;
	FILE *flog, *fif=NULL;
	clock_t start, now;
	int fif_used=0;	
	flog=fopen(train_log_file.c_str(), "ab");

	if (!fea_matrix_used) {
		if (fea_size>0) {
			fif_used=1;		//this means that feature matrix file was not set and features are used
		}
	}

	//sprintf(log_name, "%sLog", train_log_file.c_str());
	counter=train_cur_pos;
	//featureIndexer = new FeatureIndexer("c:/temp/featureCount");
	RNN* preModel = NULL;

	while (1) { //iteration loop
		//save model structure for one iteration trainig. Do not change alpha
		//as it is set by the master RNN setting
		if (one_iter == true)
		{
			//saveWeights();
		}
		else
		{
			if (alpha_divide)
			{
				alpha /= 2;
			}
		}
		printf("Iter: %3d\tAlpha: %f\t   ", iter, alpha);
		fflush(stdout);

		if (bptt>0) {
			for (a=0; a<bptt+bptt_block; a++) {
				bptt_history[a]=0;
			}
		}
		for (a=0; a<MAX_NGRAM_ORDER; a++) {
			history[a]=0;
		}

		//TRAINING PHASE
		netFlush();

		//fi=fopen(train_file.c_str(), "rb");
		if (fif_used) {
			fif=fopen(fea_file.c_str(), "rb");
			fread(&fea_size, sizeof(fea_size), 1, fif);
		}
		if (fea_matrix_used) {
			for (a=0; a<fea_size; a++) {
				neuf[a].ac=0;
			}
		}

		last_word=0;

		//comment the unimportant feature
		//if (counter>0) {
		//	for (a=0; a<counter; a++) {
		//		word=readWordIndex(fi, wordIndexed);	//this will skip words that were already learned if the training was interrupted
		//	}
		//}
		start=clock();
		llogpTrain = logpTrain;
		logpTrain = 0;
		ifstream trainIS(train_file);
		string line;
		while(getline(trainIS, line)) { //line loop
			if(line.empty()) continue; //skip empty lines
			event.clear();
			vector<string> wordStrs;
			Utils::Tokenize(line, event, "\t");
			Utils::Tokenize(event[0], wordStrs, " ");
			if(wordStrs.size() == 0) continue;
			if(wordIndexed) {
				wordStrs.push_back("0");
			} else {
				wordStrs.push_back("</s>");
			}
			for(int i = 0; i < wordStrs.size(); i++) {
				counter++;
				if ((counter%10000)==0) {
					now=clock();
					if (train_words>0) {
						//printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f ", 
						//	13, iter, alpha, -logpTrain/log10((double)2)/counter, counter/(real)train_words*100, counter/((double)(now-start)/1000.0));
						fprintf(flog, "%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f\n", 
							13, iter, alpha, -logpTrain/log10((double)2)/counter, counter/(real)train_words*100, counter/((double)(now-start)/1000.0));
						fflush(flog);
					} else {
						//printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %dK", 
						//13, iter, alpha, -logpTrain/log10((double)2)/counter, counter/1000);
					}
					fflush(stdout);
				}
				word = getWordIndex(wordStrs[i], wordIndexed);
				//word=readWordIndex(fi, wordIndexed);     //read next word
				//if(word == -2) break;
				//temporarily commented
				//if (!fea_matrix_used) {
				//	for (a=0; a<fea_size; a++) {	//read extra features if fea_size>0
				//		fread(&fl, sizeof(fl), 1, fif);
				//		neuf[a].ac=fl;
				//	}
				//} else {		//fea matrix file is used
				//	////////////////
				//	if (counter!=1)
				//		updateFeatureVector(last_word);
				//}

				computeNet(last_word, word, false);      //compute probability distribution

				//if (feof(fi)) break;        //end of file: test on validation data, iterate till convergence

				if (word!=-1) {
					if(vocab->superClassSize > 1) {
						logpTrain+=log10(neu2[vocab->vocabSize()+ vocab->classSize + vocab->getSuperClass(word)].ac * neu2[vocab->vocabSize()+vocab->getClass(word)].ac * neu2[word].ac);
					} else {
						logpTrain+=log10(neu2[vocab->vocabSize()+vocab->getClass(word)].ac * neu2[word].ac);
					}
				}

				//
				if (bptt>0) {		//shift memory needed for bptt to next time step
					for (a=bptt+bptt_block-1; a>0; a--) {
						bptt_history[a]=bptt_history[a-1];
					}
					bptt_history[0]=last_word;

					for (a=bptt+bptt_block-1; a>0; a--) {
						for (b=0; b<layer1_size; b++) {
							bptt_hidden[a*layer1_size+b].ac=bptt_hidden[(a-1)*layer1_size+b].ac;
							bptt_hidden[a*layer1_size+b].er=bptt_hidden[(a-1)*layer1_size+b].er;
						}
					}

					for (a=bptt+bptt_block-1; a>0; a--) {
						for (b=0; b<fea_size; b++) {
							bptt_fea[a*fea_size+b].ac=bptt_fea[(a-1)*fea_size+b].ac;
						}
					}
				}
				//
				learnNet(last_word, word);

				//feature indexer
				//featureIndexer->increaseFeatureCount("w", history, word, direct_order, 1);
				//featureIndexer->increaseFeatureCount("c", history, vocab->getClass(word), direct_order, 1);	

				copyHiddenLayerToInput();

				if (last_word!=-1) {
					neu0[last_word].ac=0;  //delete previous activation
				}
				last_word=word;

				for (a=MAX_NGRAM_ORDER-1; a>0; a--) {
					history[a]=history[a-1];
				}
				history[0]=last_word;
				//word == 0 means </s>
				if (independent && (word==0)) {
					netReset();
				}
			} //end for(int i = 0; i < wordStrs.size(); i++) 
		} //end while(getline(trainIS, line))

		//featureIndexer->saveFeatureCount("c:/temp/featureCount", 1);
		//exit(0);

		trainIS.close();

		if (fif_used) {
			fclose(fif);
		}		

		now=clock();
		printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f   ", 
			13, iter, alpha, -logpTrain/log10((double)2)/counter, counter/((double)(now-start)/1000.0));
		fprintf(flog, "%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f\n", 
			13, iter, alpha, -logpTrain/log10((double)2)/counter, counter/((double)(now-start)/1000.0));
		fflush(flog);

		evaluateNet(flog);

		counter=0;
		train_cur_pos=0;
		iter++;
		//
		//RNN client 
		if (one_iter == true) {
			//double tolerance = 1.01;
			//if (logpValid < llogpValid * tolerance) //allow some space to go worse
			//{
			//	fprintf(flog, "restore to previous model as current logprob %f is less than previous logprob %f with tolerance %f\n", logpValid, llogpValid, tolerance);
			//	restoreWeights();
			//	logpValid = llogpValid;				
			//}
			break;
		} else //normal RNN train
		{
			//ave the rnn weights if the logp in this iteration is larger than the 
			//previous one, otherwise, load the previous rnn weights
			if (logpValid < llogpValid) {
				if(preModel != NULL) {
					copyNet(*preModel);
				}
				//restoreWeights();
			} else {
				if(preModel == NULL) {
					preModel = new RNN(*this, true);
				} else {
					preModel->copyNet(*this);
				}
				//saveWeights();
			}
			std::stringstream sstm;
			sstm << train_log_file.substr(0, train_log_file.length()-3) << "Iter" << iter;	
			saveNet(sstm.str(), true);
			if (logpValid * min_improvement < llogpValid) {
				if (alpha_divide == 0) {
					alpha_divide = 1;
				} else {
					break;
				}
			}

			if (maxIter > 0 && iter == maxIter) {
				break;
			}
		}
	} //end iteration
	//write a file to indicate training is done!
	if (one_iter == true) {
		iter--;
		//do not write done message here, write after model is saved!
		//fprintf(flog, "%s%i\n\n", DONE_STR.c_str(), iter);
	} else {
		if(preModel != NULL) {
			delete preModel;
		}
	}
	fclose(flog);
}

//Evaluate RNN model against validation data and update llogpValid
//and logpValid
double RNN::evaluateNet(FILE *flog) {
	//VALIDATION PHASE
	netFlush();

	//FILE *fi=fopen(valid_file.c_str(), "rb");
	ifstream validIS(valid_file);
	if (!validIS.good()) {
		printf("Valid file not found\n");
		exit(1);
	}
	string line;

	//to add later
	/*if (fif_used) {
	fif=fopen(fea_valid_file, "rb");
	fread(&fea_size, sizeof(fea_size), 1, fif);
	}
	if (fea_matrix_used) {
	for (a=0; a<fea_size; a++) {
	neuf[a].ac=0;
	}
	}*/

	if (flog==NULL) {
		printf("Cannot open log file\n");
		exit(1);
	}

	llogpValid = logpValid;
	logpValid = 0;
	int last_word=0;
	int wordcn=0;
	int unk_idx = vocab->getWordId("<unk>");
	//assert(unk_idx != -1);
	while(getline(validIS, line)) { //line loop
		if(line.empty()) continue; //skip empty lines
		event.clear();
		vector<string> wordStrs;
		Utils::Tokenize(line, event, "\t");
		Utils::Tokenize(event[0], wordStrs, " ");
		if(wordStrs.size() == 0) continue;
		wordStrs.push_back("</s>");
		for(int i = 0; i < wordStrs.size(); i++) {				
			int word = getWordIndex(wordStrs[i], false);
			//to add it later
			//if (!fea_matrix_used) {
			//	for (a=0; a<fea_size; a++) {	//read extra features
			//		fread(&fl, sizeof(fl), 1, fif);
			//		neuf[a].ac=fl;
			//	}
			//} else {		//fea matrix file is used
			//	////////////////
			//	updateFeatureVector(last_word);
			//}
			if (word == -2) break;
			//replace oov with <unk>
			if(word == -1) {
				word = unk_idx; //unknown word is replaced with <unk>
			}
			computeNet(last_word, word, false);      //compute probability distribution
			//if (feof(fi)) break;              //end of file: report LOGP, PPL

			if (word!=-1) {
				if(vocab->superClassSize > 1) {
					logpValid+=log10(neu2[vocab->vocabSize()+ vocab->classSize + vocab->getSuperClass(word)].ac * neu2[vocab->vocabSize()+vocab->getClass(word)].ac * neu2[word].ac);
				} else {
					logpValid+=log10(neu2[vocab->vocabSize()+vocab->getClass(word)].ac * neu2[word].ac);
				}
				wordcn++;
			}

			/*if (word!=-1)
			fprintf(flog, "%d\t%f\t%s\n", word, neu2[word].ac, vocab[word].word);
			else
			fprintf(flog, "-1\t0\t\tOOV\n");*/

			copyHiddenLayerToInput();

			if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

			last_word=word;

			for (int a=MAX_NGRAM_ORDER-1; a>0; a--) {
				history[a]=history[a-1];
			}
			history[0]=last_word;
			//word == 0 means </s>
			if (independent && (word==0)) {
				netReset();
			}
		} //end for(int i = 0; i < wordStrs.size(); i++)
	} //end while(getline(validIS, line))
	validIS.close();
	//saveNet("//fbl/nas/HOME/zhihuang/RNNParaExp/smallSingle/modelRnnTemp");
	fprintf(flog, "iter: %d alpha: %f\n", iter, alpha);
	fprintf(flog, "previous valid log probability: %f pplx: %f\n", llogpValid, exp_10(-llogpValid/(real)wordcn));
	fprintf(flog, "current valid log probability: %f pplx: %f\n", logpValid, exp_10(-logpValid/(real)wordcn));
	fprintf(flog, "current/previous log probability: %f\n", logpValid/llogpValid);
	fflush(flog);
	//printf("VALID entropy: %.4f\n", -logpValid/log10((double)2)/wordcn);
	return logpValid;
}

//test RNN and write to log file. OOVs are replaced with <unk> if replace is true.
//In previous version, this method allows an aditional lm file for interpolation
void RNN::testNet(string test_file, bool replace, double unk_penalty, FILE* logger, int debug_mode)
{
	//wordcn is for regular and <unk> tokens.
	//word_unk is for <unk> tokens, word_oov is the real OOV tokens
	int a, word; 
	FILE *fif=NULL;
	int fif_used=0;
	int last_word=0;					//last word = end of sentence
	double logp=0; //log prob for wordcn
	double logpUnk=0;	
	int wordcn=0;
	int word_unk=0;
	int word_oov=0;
	int numSen=0;

	if (!fea_matrix_used) {
		if (fea_size>0) {
			fif_used=1;		// this means that feature matrix file was not set and features are used
		}
	}
	//restoreNet();

	//TEST PHASE
	//netFlush();

	//fi=fopen(test_file.c_str(), "rb");
	//flog=stdout;
	if (debug_mode>0) {
		fprintf(logger, "Index   P(NET)          Word\n");
		fprintf(logger, "----------------------------------\n");
	}

	if (fif_used) {
		if (strlen(fea_file.c_str())==0) {
			printf("Feature file for the test data is needed to evaluate this model (use -features <FILE>)\n");
			exit(1);
		}
		fif=fopen(fea_file.c_str(), "rb");
		fread(&a, sizeof(a), 1, fif);
		if (a!=fea_size) {
			printf("Mismatch between feature vector size in model file and feature file (model uses %d features, in %s found %d features)\n", fea_size, fea_file.c_str(), a);
			exit(1);
		}
	}
	if (fea_matrix_used) {
		for (a=0; a<fea_size; a++) {
			neuf[a].ac=0;
		}
	}

	/*if (debug_mode>1)	{
	fprintf(flog, "Index   P(NET)          Word\n");
	fprintf(flog, "----------------------------------\n");
	}*/

	copyHiddenLayerToInput();

	if (bptt>0) {
		for (a=0; a<bptt+bptt_block; a++) {
			bptt_history[a]=0;
		}
	}
	for (a=0; a<MAX_NGRAM_ORDER; a++) {
		history[a]=0;
	}
	if (independent) {
		netReset();
	}

	int unk_idx = vocab->getWordId("<unk>");
	if(unk_idx == -1 && replace == true) {
		replace = false;
		printf("replace changed to false due to no <unk> token\n");
	}
	//assert(unk_idx != -1);
	int endTagIdx = vocab->getWordId("</s>");
	//cout << "unk_idx:" << unk_idx << " endTagIdx:" << endTagIdx << endl;
	//unk_idx:1 endTagIdx:0
	ifstream testIS(test_file);
	string line;
	while(getline(testIS, line)) { //line loop
		if(line.empty()) continue; //skip empty lines
		event.clear();
		vector<string> wordStrs;
		Utils::Tokenize(line, event, "\t");
		Utils::Tokenize(event[0], wordStrs, " ");
		if(wordStrs.size() == 0) continue;
		wordStrs.push_back("</s>");

		for(int i = 0; i < wordStrs.size(); i++) {				
			word = getWordIndex(wordStrs[i], false);

			//while (1) { //read one word per iteration for normal text and </s>
			//word=readWordIndex(fi, false);		//read next word
			//if(word == -2) break;
			if(replace && (word == -1)) {
				word = unk_idx; //unknown word is replaced with <unk>
			}
			//if (fif_used) {
			//	for (a=0; a<fea_size; a++) {	//read extra features
			//		fread(&fl, sizeof(fl), 1, fif);
			//		neuf[a].ac=fl;
			//	}
			//}
			//if (fea_matrix_used) {
			//	updateFeatureVector(last_word);
			//}
			// compute probability distribution. Nothing is done if word==-1
			computeNet(last_word, word, false);		
			//if (feof(fi)) {
			//	break;				// end of file: report LOGP, PPL
			//}
			double lp = 0;
			if (word!=-1) { //regular token or <unk> token
				if(vocab->superClassSize > 1) {
					lp = log10(neu2[vocab->vocabSize()+ vocab->classSize + vocab->getSuperClass(word)].ac * neu2[vocab->vocabSize()+vocab->getClass(word)].ac * neu2[word].ac);
				} else {
					lp = log10(neu2[vocab->vocabSize()+vocab->getClass(word)].ac * neu2[word].ac);
				}
				if(word == unk_idx) {
					word_unk++;
					lp += unk_penalty;
					logpUnk += lp;
				}
				logp += lp;
				wordcn++;				
			} else { //OOV
				word_oov++;
			}

			if (debug_mode>0) {				
				if (word!=-1) {
					fprintf(logger, "%d\t%.10f\t%s", word, exp_10(lp), vocab->getWordStr(word).c_str());
				} else {
					fprintf(logger, "-1\t0\t\tOOV");
				}
				fprintf(logger, "\n");
			}

			copyHiddenLayerToInput();

			if (last_word!=-1) {
				neu0[last_word].ac=0;  //delete previous activation
			}

			last_word=word;

			for (a=MAX_NGRAM_ORDER-1; a>0; a--) {
				history[a]=history[a-1];
			}
			history[0]=last_word;

			if (independent && (word==0)) {
				netReset(); //word == 0 means </s>
				numSen++;
			}
		} //end for(int i = 0; i < wordStrs.size(); i++)
	} //end while(getline(testIS, line)) 
	testIS.close();
	//fclose(fi);
	if (fif_used) {
		fclose(fif);
	}

	//write to log file
	fprintf(logger, "\ntest log probability: %f\n", logp);
	fprintf(logger, "num lines: %i\n", numSen);
	fprintf(logger, "num words: %i\n", (wordcn-numSen));
	fprintf(logger, "perlexity (including <unk>):%f\n",exp_10(-logp/(real)wordcn));
	real pplx = exp_10(-(logp-logpUnk)/(real)(wordcn-word_unk));
	fprintf(logger, "perlexity (excluding <unk>):%f\n",pplx);
	real unkRate = (real)word_unk/(wordcn-numSen)*100;
	fprintf(logger, "<unk> (%%) %f \n", unkRate);
	fprintf(logger, "OOV words: %d\n", word_oov); 
	fflush(logger);
}

bool RNN::loadTarget(FILE *f) {
	float fl;
	for (int a=0; a<fea_size; a++) {   
		if (fread(&fl, sizeof(fl), 1, f) != 1) { // reached eof
			return false;
		}
		neuf[a].ac=fl;
	}
	return true;
}

//read string from stdin and populate prefix with word ids
void RNN::get_prefix(vector<int> &prefix) {
	prefix.clear();
	cout << "Enter sentence prefix" << endl;
	char buff[10000];
	if (!gets(buff))
		exit(0);
	//const int oov = vocab->getWordId("<unk>");
	char *w = strtok(buff, " \t\n");
	while (w) {
		prefix.push_back(vocab->getWordId(w));
		w = strtok(NULL, " \t\n");
	}
}

//sample elements according to their probabilities
int RNN::sample_head(vector<pair<double, int> > &probs, int head) {
	head = (head>probs.size())?(probs.size()):(head);
	sort(probs.begin(), probs.end());
	reverse(probs.begin(), probs.end());
	double tot = 0;
	for (int i=0; i<head; i++) {
		tot += probs[i].first;
	}
	real f = Utils::random(0, 1), g=0;
	for (int i=0; i<head; i++) {
		g += probs[i].first/tot;
		if (g >= f) {
			return probs[i].second;
		}
	}
	if (g>0.999999 && g<1.00000001) {  // essentially added to 1; rounding error
		return probs.back().second;
	}
	cout << "Sampling distribution did not sum to 1" << endl;
	cout << "Target: " << f << "; number of choices: " << probs.size() << "; sum: " << g << endl;
	exit(1);
}


//generate and print geneated words. head  == -1 means allows any classes, otherwise, generate
//words from the top probable head classes
void RNN::testGen(int gen, int head, bool generate_interactive, string outFile)
{
	int i, word, cla, last_word, wordcn, c, b, a=0;
	real sum, val;
	int unk_idx = vocab->getWordId("<unk>");
	assert(unk_idx != -1);
	ofstream writer(outFile, ios::out);

	int fif_used = 0;
	if (!fea_matrix_used && fea_size>0) fif_used=1;		//this means that feature matrix file was not set and features are used

	//restoreNet();

	FILE *fif=NULL;
	if (fif_used) {
		if (strlen(fea_file.c_str())==0) {
			printf("Feature file with targets is needed to generate with this model (use -features <FILE>)\n");
			exit(1);
		}
		fif=fopen(fea_file.c_str(), "rb");
		fread(&a, sizeof(a), 1, fif);
		if (a!=fea_size) {
			printf("Mismatch between feature vector size in model file and feature file (model uses %d features, in %s found %d features)\n", fea_size, fea_file.c_str(), a);
			exit(1);
		}
	}

	if (fea_matrix_used && fif_used) {
		cout << "cannot use both a feature matrix and feature file\n";
		exit(1);
	}

	if (fif_used && generate_interactive) {
		cout << "cannot generate interactively with a model requiring an external feature file" << endl;
		exit(1);
	}

	word=0;
	last_word=0;					// last word = end of sentence
	wordcn=0;
	copyHiddenLayerToInput();
	if (fea_matrix_used) {
		for (a=0; a<fea_size; a++) {
			neuf[a].ac=0;
		}
	}

	if (fif_used && !loadTarget(fif)) { // no target vectors
		cout << "empty target vector file\n";
		exit(1);
	}

	vector<int> prefix;
	if (generate_interactive) {
		get_prefix(prefix);
	}
	int prefix_pos = 0;

	std::stringstream ss;
	bool unkSampled = false;
	while (wordcn<gen) {
		// if generating towards a target, use a different updateFeatureVector function. For a constant target, do nothing.
		if (fea_matrix_used) 
			updateFeatureVector(last_word);
		//class prob estimation is okay, but the words prob are computed based on class of </s> (index being 0)
		computeNet(last_word, 0, true);		// compute probability distribution

		if (vocab->classSize == 0) { // not using classes
			cla = 0;
		} else {  // sample from class distribution //why no direc connections to classes???
			vector<pair<double, int> > class_probs;
			for (i=vocab->vocabSize(); i<layer2_size; i++) 
				class_probs.push_back(pair<double, int>(neu2[i].ac, i-vocab->vocabSize()));
			cla = sample_head(class_probs, (head==-1)?(class_probs.size()):(head));
		}
		assert(cla>=0 && cla<vocab->classSize);

		//		if (cla>class_size-1) cla=class_size-1;
		//		if (cla<0) cla=0;

		//
		// !!!!!!!!  THIS WILL WORK ONLY IF CLASSES ARE CONTINUALLY DEFINED IN VOCAB !!! (like class 10 = words 11 12 13; not 11 12 16)  !!!!!!!!
		// forward pass 1->2 for words
		int start = vocab->getWordIdFromClass(cla, 0);
		//int end = vocab->getWordIdFromClass(cla, vocab->numWordsInClass(cla));
		int end = start + vocab->numWordsInClass(cla);
		for (c=start; c<end; c++) {
			neu2[c].ac=0;
		}
		matrixXvector(neu2, neu1, syn1, layer1_size, start, end, 0, layer1_size, 0);

		//fea to out word
		if (fea_size>0)	matrixXvector(neu2, neuf, synfo, fea_size, start, end, 0, fea_size, 0);

		//apply direct connections to words
		if (last_word!=-1) {
			if (direct_size>0) {
				//ngram order (0, 1, 2 etc) to its feature index
				vector<unsigned long long> hash;	//this will hold pointers to syn_d that contains hash parameters
				//to add the super class part!!
				getMEFeaIds(hash, 1, cla);

				for (a = start; a < end; a++)
				{
					for (b=0; b<hash.size(); b++) {					
						neu2[a].ac+=syn_d[hash[b]];	
						hash[b]++;
						hash[b]=hash[b]%direct_size;
					}
				}

				/*string ngram = "ngram#word|";
				for (int i = 0; i < direct_order; i++) {
				if(i == 1) {
				ngram += history[i-1];
				} else if (i > 1) {
				ngram +="-" + history[i-1];
				}
				for (a = start; a < end; a++) {
				std::stringstream sstm;
				sstm << ngram << "#" << a;	
				string f = sstm.str();
				long long featureId = featureIndexer->getFeaId(f, false); 
				if(featureId != -1) {
				neu2[a].ac+=syn_d[featureId];
				}
				}
				}*/
			}
		}

		//activation 2   --softmax on words
		sum=0;
		for (a = start; a < end; a++)
		{
			if (neu2[a].ac>50) neu2[a].ac=50;  //for numerical stability
			if (neu2[a].ac<-50) neu2[a].ac=-50;  //for numerical stability
			val=exp(neu2[a].ac);
			sum+=val;
			neu2[a].ac=val;
		}
		for (a = start; a < end; a++)
		{
			neu2[a].ac/=sum;
		}
		//

		vector<pair<double, int> > word_probs;
		for (i = start; i < end; i++) {	
			word_probs.push_back(pair<double, int>(neu2[i].ac, i));
		}
		word = sample_head(word_probs, (head==-1)?(word_probs.size()):(head));
		if(word == unk_idx) {
			unkSampled = true;
		}

		if (generate_interactive) // enfore that the prefix matches...
			if (prefix_pos < prefix.size()) 
				word = prefix[prefix_pos++]++;

		assert(word>=0 && word<vocab->vocabSize());

		//		if (word>vocab_size-1) word=vocab_size-1;
		//		if (word<0) word=0;

		//printf("%s %d %d\n", vocab[word].word, cla, word);
		if (word!=0) {
			ss << vocab->getWordStr(word) << " ";
		} else {
			string sen =  ss.str();
			if(sen.length() > 0 && unkSampled == false) {
				writer << "<s> " << sen << "</s>\t1" << endl;
			}
			ss.str("");
			unkSampled = false;
		}
		copyHiddenLayerToInput();

		if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

		last_word=word;

		for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
		history[0]=last_word;

		if (independent && (word==0)) netReset();

		wordcn++;

		// if word is 0 and generating towards a target, load the new target vector
		if (fif_used && (word==0))
			if (!loadTarget(fif)) // no more taregt vectors
				break;

		if (generate_interactive && word == 0) {
			get_prefix(prefix);
			prefix_pos = 0;
		}
	} //end while (wordcn<gen)
	writer.close();
}

//save word -> its weights to hidden layer
void RNN::saveWordProjections()
{
	FILE *fo=fopen("word_projections.txt", "wb");
	int a,b;

	//restoreNet();

	fprintf(fo, "%d %d\n", vocab->vocabSize(), layer1_size);

	for (a=0; a<vocab->vocabSize(); a++) {
		fprintf(fo, "%s ", vocab->getWordStr(a).c_str());
		for (b=0; b<layer1_size; b++) {
			fprintf(fo, "%lf ", syn0[a+b*layer0_size].weight);
		}
		fprintf(fo, "\n");
	}

	fclose(fo);
}

void RNN::saveState(RNN_state &s) {
	s.neu1.resize(layer1_size);
	s.neuc.resize(layerc_size);
	s.history.resize(MAX_NGRAM_ORDER);
	copy(neu1, neu1+layer1_size, &s.neu1[0]);
	copy(neuc, neuc+layerc_size, &s.neuc[0]);
	copy(history, history+MAX_NGRAM_ORDER, &s.history[0]);
}

void RNN::setState(RNN_state &s) {
	assert(s.neu1.size() == layer1_size);
	assert(s.neuc.size() == layerc_size);
	copy(s.neu1.begin(), s.neu1.end(), neu1);
	copy(s.neuc.begin(), s.neuc.end(), neuc);
	copy(s.history.begin(), s.history.end(), history);
}

void RNN::initialize4Rescore() {
	FILE *fif=NULL;
	int fif_used=0;

	if (!fea_matrix_used) {
		if (fea_size>0) {
			fif_used=1;		// this means that feature matrix file was not set and features are used
		}
	}

	//restoreNet();

	if (fif_used) {
		if (strlen(fea_file.c_str())==0) {
			printf("Feature file for the test data is needed to evaluate this model (use -features <FILE>)\n");
			exit(1);
		}
		fif=fopen(fea_file.c_str(), "rb");
		int a;
		fread(&a, sizeof(a), 1, fif);
		if (a!=fea_size) {
			printf("Mismatch between feature vector size in model file and feature file (model uses %d features, in %s found %d features)\n", fea_size, fea_file.c_str(), a);
			exit(1);
		}
	}
	if (fea_matrix_used) {
		for (int a=0; a<fea_size; a++) neuf[a].ac=0;
	}
	netReset();
}

double RNN::pathExtend(string &addword, const vector<float> &features, double unk_penalty) {

	// restore the old state; run the network forwards to compute the probability of the words; store the resulting hidden state and
	// return the log-probability of the words in the extension

	const int unk_idx = vocab->getWordId("<unk>");
	assert(unk_idx != -1);

	if (fea_matrix_used) {
		cerr << "Lattice rescoring does not support feature matrix\n";
		cerr << "Please pass features in explicitly\n";
		exit(1);
	}

	int last_word = history[0];

	double lp = 0.0;
	copyHiddenLayerToInput();

	int word = vocab->getWordId(addword);
	if(word==-1) {
		word = unk_idx;
	}
	if (features.size()) {
		assert(features.size() == fea_size);
		for (int a=0; a<fea_size; a++) 	// set the  features
			neuf[a].ac=features[a];
	}

	computeNet(last_word, word, false);		//compute probability distribution
	//to handle super class
	lp+=log(neu2[vocab->getClass(word)+vocab->vocabSize()].ac * neu2[word].ac);
	if (word == unk_idx)
		lp += unk_penalty;  // <unk> is a type not a token; unk_penalty should be about -log(#unknown words) to compensate

	copyHiddenLayerToInput();

	if (last_word!=-1) {
		neu0[last_word].ac=0;  //delete previous activation
	}
	last_word=word;
	for (int a=MAX_NGRAM_ORDER-1; a>0; a--) {
		history[a]=history[a-1];
	}
	history[0]=last_word;

	return lp;
}

//Merge two RNN models
void RNN::add(RNN &model)
{
	ensureSameNetStructure(model);
	//do we need this??
	//Hidden layer activation
	for (int a = 0; a < layer1_size; a++)
	{
		neu1[a].ac += model.neu1[a].ac;
	}

	//Weights 0->1
	for (int b = 0; b < layer1_size; b++)
	{
		for (int a = 0; a < layer0_size; a++)
		{
			syn0[a + b * layer0_size].weight += model.syn0[a + b * layer0_size].weight;
		}
	}

	//Weights fea->1
	for (int b = 0; b < layer1_size; b++)
	{
		for (int a = 0; a < fea_size; a++)
		{
			synf[a + b * fea_size].weight += model.synf[a + b * fea_size].weight;
		}
	}

	//Weights fea->out
	for (int b = 0; b < layer2_size; b++)
	{
		for (int a = 0; a < fea_size; a++)
		{
			synfo[a + b * fea_size].weight += model.synfo[a + b * fea_size].weight;
		}
	}

	if (layerc_size > 0)
	{
		//nWeights 1->c
		for (int b = 0; b < layerc_size; b++)
		{
			for (int a = 0; a < layer1_size; a++)
			{
				syn1[a + b * layer1_size].weight += model.syn1[a + b * layer1_size].weight;
			}
		}

		//Weights c->2
		for (int b = 0; b < layer2_size; b++)
		{
			for (int a = 0; a < layerc_size; a++)
			{
				sync[a + b * layerc_size].weight += model.sync[a + b * layerc_size].weight;
			}
		}
	}
	else
	{
		//Weights 1->2
		for (int b = 0; b < layer2_size; b++)
		{
			for (int a = 0; a < layer1_size; a++)
			{
				syn1[a + b * layer1_size].weight += model.syn1[a + b * layer1_size].weight;
			}
		}
	}


	//Direct connections
	long long aa;
	for (aa = 0; aa < direct_size; aa++)
	{
		syn_d[aa] += model.syn_d[aa];
	}

	if (fea_matrix_used)
	{

		//Feature matrix
		for (int b = 0; b < vocab->vocabSize(); b++)
		{
			for (int a = 0; a < fea_size; a++)
			{
				fea_matrix[a + b * fea_size] += model.fea_matrix[a + b * fea_size];
			}
		}
	}
}

void RNN::divide(int num)
{
	//do we need this??
	//Hidden layer activation
	for (int a = 0; a < layer1_size; a++)
	{
		neu1[a].ac /= num;
	}

	//Weights 0->1
	for (int b = 0; b < layer1_size; b++)
	{
		for (int a = 0; a < layer0_size; a++)
		{
			syn0[a + b * layer0_size].weight /= num;
		}
	}

	//Weights fea->1
	for (int b = 0; b < layer1_size; b++)
	{
		for (int a = 0; a < fea_size; a++)
		{
			synf[a + b * fea_size].weight /= num;
		}
	}

	//Weights fea->out
	for (int b = 0; b < layer2_size; b++)
	{
		for (int a = 0; a < fea_size; a++)
		{
			synfo[a + b * fea_size].weight /= num;
		}
	}

	if (layerc_size > 0)
	{
		//nWeights 1->c
		for (int b = 0; b < layerc_size; b++)
		{
			for (int a = 0; a < layer1_size; a++)
			{
				syn1[a + b * layer1_size].weight /= num;
			}
		}

		//Weights c->2
		for (int b = 0; b < layer2_size; b++)
		{
			for (int a = 0; a < layerc_size; a++)
			{
				sync[a + b * layerc_size].weight /= num;
			}
		}
	}
	else
	{
		//Weights 1->2
		for (int b = 0; b < layer2_size; b++)
		{
			for (int a = 0; a < layer1_size; a++)
			{
				syn1[a + b * layer1_size].weight /= num;
			}
		}
	}


	//Direct connections
	long long aa;
	for (aa = 0; aa < direct_size; aa++)
	{
		syn_d[aa] /= num;
	}

	if (fea_matrix_used)
	{

		//Feature matrix
		for (int b = 0; b < vocab->vocabSize(); b++)
		{
			for (int a = 0; a < fea_size; a++)
			{
				fea_matrix[a + b * fea_size] /= num;
			}
		}
	}
}

//preAveModel provides momentum info
void RNN::update(RNN &averageRNN, RNN &preAveModel, double learningRate, double beta)
{
	double low_com = 1; //0.5
	ensureSameNetStructure(averageRNN);

	//Hidden layer activation
	//for (int a = 0; a < layer1_size; a++)
	//{
	//    neu1[a].ac /= num;
	//}

	//Weights 0->1
	for (int b = 0; b < layer1_size; b++)
	{
		for (int a = 0; a < layer0_size; a++)
		{
			double d = averageRNN.syn0[a + b * layer0_size].weight - syn0[a + b * layer0_size].weight;
			syn0[a + b * layer0_size].weight += low_com* d * learningRate - beta * syn0[a + b * layer0_size].weight ;
			//syn1[a+b*L1].weight += batch_learning_rate*d/(double)size -beta3*syn1[a+b*L1].weight+ momentum*pre_dsyn1[a+b*L1].weight;
		}
	}

	//Weights fea->1
	//for (int b = 0; b < layer1_size; b++)
	//{
	//    for (int a = 0; a < feaSize; a++)
	//    {
	//        synf[a + b * feaSize].weight /= num;
	//    }
	//}

	//Weights fea->out
	//for (int b = 0; b < layer2_size; b++)
	//{
	//    for (int a = 0; a < feaSize; a++)
	//    {
	//        synfo[a + b * feaSize].weight /= num;
	//    }
	//}

	if (layerc_size > 0)
	{
		//nWeights 1->c
		//for (int b = 0; b < layerc_size; b++)
		//{
		//    for (int a = 0; a < layer1_size; a++)
		//    {
		//        syn1[a + b * layer1_size].weight /= num;
		//    }
		//}

		////Weights c->2
		//for (int b = 0; b < layer2_size; b++)
		//{
		//    for (int a = 0; a < layerc_size; a++)
		//    {
		//        sync[a + b * layerc_size].weight /= num;
		//    }
		//}
	}
	else
	{
		//Weights 1->2
		for (int b = 0; b < layer2_size; b++)
		{
			for (int a = 0; a < layer1_size; a++)
			{
				double d = averageRNN.syn1[a + b * layer1_size].weight - syn1[a + b * layer1_size].weight;
				syn1[a + b * layer1_size].weight += d * learningRate - beta * syn1[a + b * layer1_size].weight;
			}
		}
	}


	//Direct connections, no momentum is used
	long long aa;
	for (aa = 0; aa < direct_size; aa++)
	{
		double d = averageRNN.syn_d[aa] - syn_d[aa];
		syn_d[aa] += d*learningRate -beta*syn_d[aa];
	}

	//if (fea_matrix_used)
	//{

	//    //Feature matrix
	//    for (int b = 0; b < vocab.vocabSize(); b++)
	//    {
	//        for (int a = 0; a < feaSize; a++)
	//        {
	//            fea_matrix[a + b * feaSize] /= num;
	//        }
	//    }
	//}
}

//compute the difference of network and storem them to back up space
/*
void RNN::computeMomentum(RNN &prevModel)
{
ensureSameNetStructure(prevModel);
//do we need this??
//Hidden layer activation
//for (int a = 0; a < layer1_size; a++)
//{
//    neu1[a].ac += model.neu1[a].ac;
//}

//Weights 0->1
for (int b = 0; b < layer1_size; b++)
{
for (int a = 0; a < layer0_size; a++)
{
syn0b[a + b * layer0_size].weight = syn0[a + b * layer0_size].weight - prevModel.syn0[a + b * layer0_size].weight;
}
}

//Weights fea->1
//for (int b = 0; b < layer1_size; b++)
//{
//    for (int a = 0; a < feaSize; a++)
//    {
//        synf[a + b * feaSize].weight += model.synf[a + b * feaSize].weight;
//    }
//}

////Weights fea->out
//for (int b = 0; b < layer2_size; b++)
//{
//    for (int a = 0; a < feaSize; a++)
//    {
//        synfo[a + b * feaSize].weight += model.synfo[a + b * feaSize].weight;
//    }
//}

//if (layerc_size > 0)
//{
//    //nWeights 1->c
//    for (int b = 0; b < layerc_size; b++)
//    {
//        for (int a = 0; a < layer1_size; a++)
//        {
//            syn1[a + b * layer1_size].weight += model.syn1[a + b * layer1_size].weight;
//        }
//    }

//    //Weights c->2
//    for (int b = 0; b < layer2_size; b++)
//    {
//        for (int a = 0; a < layerc_size; a++)
//        {
//            sync[a + b * layerc_size].weight += model.sync[a + b * layerc_size].weight;
//        }
//    }
//}
//else
//{
//Weights 1->2
for (int b = 0; b < layer2_size; b++)
{
for (int a = 0; a < layer1_size; a++)
{
syn1b[a + b * layer1_size].weight = syn1[a + b * layer1_size].weight - prevModel.syn1[a + b * layer1_size].weight;
}
}
//}


//Direct connections
long long aa;
for (aa = 0; aa < direct_size; aa++)
{
syn_db[aa] = syn_d[aa] - prevModel.syn_d[aa];
}

//if (fea_matrix_used)
//{

//    //Feature matrix
//    for (int b = 0; b < vocab.vocabSize(); b++)
//    {
//        for (int a = 0; a < feaSize; a++)
//        {
//            fea_matrix[a + b * feaSize] += model.fea_matrix[a + b * feaSize];
//        }
//    }
//}
}
*/

void RNN::ensureSameNetStructure(RNN &model)
{
	if (file_binary != model.file_binary)
	{
		cerr <<"file type mis-matched " << file_binary << " vs. " << model.file_binary << endl;
		exit(1);
	}

	//to add
	//if (vocab.Equals(model.vocab))
	//{
	//    throw new Exception("vocab is mis-matched");
	//}

	if (layer0_size != model.layer0_size)
	{
		cerr << "model input layer size mis-matched " << layer0_size << " vs. " << model.layer0_size << endl;
		exit(1);
	}

	if (layer1_size != model.layer1_size)
	{
		cerr << "model hidden layer size mis-matched " << layer1_size << " vs. " << model.layer1_size << endl;
		exit(1);
	}

	if (layerc_size != model.layerc_size)
	{
		cerr << "model compression layer size mis-matched " << layerc_size << " vs. " << model.layerc_size << endl;
		exit(1);
	}

	if (layer2_size != model.layer2_size)
	{
		cerr << "model output layer size mis-matched " << layer2_size << " vs. " << model.layer2_size << endl;
		exit(1);
	}

	if (direct_size != model.direct_size)
	{
		cerr << "model direct size mis-matched " << direct_size << " vs. " << model.direct_size << endl;
		exit(1);
	}

	if (direct_order != model.direct_order)
	{
		cerr << "model direct order mis-matched " << direct_order << " vs. " << model.direct_order << endl;
		exit(1);
	}

	if (direct_word_size != model.direct_word_size)
	{
		cerr << "model direct word size mis-matched " << direct_word_size << " vs. " << model.direct_word_size << endl;
		exit(1);
	}

	if (direct_class_size != model.direct_class_size)
	{
		cerr << "model direct class size mis-matched " << direct_class_size << " vs. " << model.direct_class_size << endl;
		exit(1);
	}
}


