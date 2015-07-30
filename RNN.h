#ifndef _RNN_H_
#define _RNN_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include "Vocab.h"
#include "Parameters.h"
#include "FeatureIndexer.h"

using namespace std;

#define MAX_STRING 100

#pragma warning( disable:4996 )

//use float (not double) for large RNN models
typedef float real;		// doubles for NN weights
typedef float direct_t;	// doubles for ME weights; TODO: check why floats are not enough for RNNME (convergence problems)

struct neuron {
	real ac;		//actual value stored in neuron
	real er;		//error value in neuron, used by learning algorithm
};

struct synapse {
	real weight;	//weight of synapse
};

const unsigned int PRIMES[]={108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
	407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
	782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
const unsigned int PRIMES_SIZE=sizeof(PRIMES)/sizeof(PRIMES[0]);

const int MAX_NGRAM_ORDER=20;
const string DONE_STR = "Done with RNN slave training iter ";

struct RNN_state {
	vector<neuron> neu1;
	vector<neuron> neuc;
	vector<int> history;
};

#define END_SENT_ID 0

class RNN {
public:
	int version;
	string train_file;
	string valid_file;
	string fea_file;
	string fea_valid_file;
	string fea_matrix_file;
	double feature_gamma;
	int rand_seed;
	bool random_start;
	
	bool file_binary;
	real gradient_cutoff;
	real starting_alpha;
	real alpha;	
	bool alpha_divide;
	double logpValid;
	double llogpValid;
	double logpTrain;
	double llogpTrain;
	float min_improvement;
	int iter;
	bool one_iter;
	int maxIter;
	int train_words;
	int train_cur_pos;
	int counter;
	real beta;

	int layer0_size;
	int layer1_size;
	int layerc_size;
	int layer2_size;

	long long direct_size;
	int direct_order;
	int direct_word_size;
	int direct_class_size;
	vector<int> context_ids;
	int history[MAX_NGRAM_ORDER];
	vector<string> event;

	int bptt;
	int bptt_block;
	int *bptt_history;
	neuron *bptt_hidden;
	neuron *bptt_fea;
	struct synapse *bptt_syn0;
	struct synapse *bptt_synf;
		
	int fea_size;
	real *fea_matrix;	//this will be used for the second way how to add features into RNN: just matrix W*T will be specified, where W=number of words (vocab_size) and T=number of topics (fea_size)
	bool fea_matrix_used;

	bool independent;

	struct neuron *neu0;		//neurons in input layer
	struct neuron *neuf;		//features in input layer
	struct neuron *neu1;		//neurons in hidden layer
	struct neuron *neuc;		//neurons in hidden layer
	struct neuron *neu2;		//neurons in output layer

	struct synapse *syn0;		//weights between input and hidden layer
	struct synapse *synf;		//weights between features and hidden layer
	struct synapse *synfo;		//weights between features and output layer
	struct synapse *syn1;		//weights between hidden and output layer (or hidden and compression if compression>0)
	struct synapse *sync;		//weights between hidden and compression layer
	direct_t *syn_d;			//direct parameters between input and output layer (similar to Maximum Entropy model parameters)

	//backup used in training:
	/*struct neuron *neu0b;
	struct neuron *neufb;
	struct neuron *neu1b;
	struct neuron *neucb;
	struct neuron *neu2b;

	struct synapse *syn0b;
	struct synapse *synfb;
	struct synapse *synfob;
	struct synapse *syn1b;
	struct synapse *syncb;
	direct_t *syn_db;*/

	Vocab *vocab;

	//backup used in n-bset rescoring:
	//struct neuron *neu1b2;	
	//FeatureIndexer *featureIndexer;

public:
	//int debug_mode;

	RNN(Parameters parameters, bool useRandom);
	RNN(string rnnlm_file, bool includeVocab);
	RNN(RNN &model, bool includeVocab);
	void copyHead(RNN &model, bool includeVocab);
	void copyNet(RNN &model);

	~RNN()		//destructor, deallocates memory
	{		
		if (neu0!=NULL) {
			if (fea_matrix!=NULL) free(fea_matrix);

			free(neu0);
			if (neuf!=NULL) free(neuf);
			free(neu1);
			if (neuc!=NULL) free(neuc);
			free(neu2);

			free(syn0);
			if (synf!=NULL) free(synf);
			if (synfo!=NULL) free(synfo);
			free(syn1);
			if (sync!=NULL) free(sync);
						
			if (syn_d!=NULL) {
				free(syn_d);
			}
			//if (syn_db!=NULL) free(syn_db);
			////
			//free(neu0b);
			//if (neufb!=NULL) free(neufb);
			//free(neu1b);
			//if (neucb!=NULL) free(neucb);
			//free(neu2b);

			//free(neu1b2);

			//free(syn0b);
			//if (synfb!=NULL) free(synfb);
			//if (synfob!=NULL) free(synfob);
			//free(syn1b);
			//if (syncb!=NULL) free(syncb);
			
			//vocab may be reference by multiple RNN instances
			if(vocab != NULL) {
				delete(vocab);
			}
			
			if (bptt_history!=NULL) free(bptt_history);
			if (bptt_hidden!=NULL) free(bptt_hidden);
			if (bptt_fea!=NULL) free(bptt_fea);
			if (bptt_syn0!=NULL) free(bptt_syn0);
			if (bptt_synf!=NULL) free(bptt_synf);
			//todo: free bptt variables too

			/*if(featureIndexer != NULL) {
				delete(featureIndexer);
			}*/
		}
	}

	real exp_10(real num);
	//real random(real min, real max);
	//bool readLine(vector<string> &strs, ifstream &in); 
	//void readWord(char *word, FILE *fin);
	//int readWordIndex(FILE *fin, bool wordIndexed);
	int getWordIndex(string &str, bool wordIndexed);

	//void saveWeights();			//saves current weights and unit activations
	//void restoreWeights();		//restores current weights and unit activations from backup copy
	void saveContext();
	void restoreContext();
	void saveContext2();
	void restoreContext2();
	void initNet(bool random_start);
	void saveNet(string model_file, bool includeVocab);
	
	void netFlush();
	void netReset();    //will erase just hidden layer state + bptt history + maxent history (called at end of sentences in the independent mode)

	void getMEFeaIds(vector<unsigned long long> &hash, int type, int parentIndex);
	void computeNet(int last_word, int word, bool generate);
	void learnNet(int last_word, int word);
	void copyHiddenLayerToInput();
	void updateFeatureVector(int w);
	void trainNet(bool wordIndexed, string train_log_file);
	double evaluateNet(FILE *log);
	void testNet(string test_file, bool replace, double unk_penalty, FILE* logger, int debug_mode);
	void testGen(int gen, int head, bool generate_interactive, string outFile);
	bool loadTarget(FILE *f);
	void get_prefix(vector<int> &prefix);
	int sample_head(vector<pair<double, int> > &probs, int head);

	void saveWordProjections();

	void matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type);

	int feaSize() {return fea_size;}
	void saveState(RNN_state &s);
	void setState(RNN_state &s);
	void initialize4Rescore();
	double pathExtend(string &addword, const vector<float> &features, double unk_penalty);
	
	void add(RNN &model);
	void divide(int num);
	void update(RNN &averageRNN, RNN &preAveModel, double learningRate, double beta);
	//void computeMomentum(RNN &prevModel);
	void ensureSameNetStructure(RNN &model);
};

#endif
