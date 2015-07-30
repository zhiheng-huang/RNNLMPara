#pragma once
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include "Vocab.h"

using namespace std;

class FeatureIndexer
{
public:
	unsigned long featureSize;
	unordered_map<string, vector<pair<int, unsigned long> > > contextLabelFId;
	//map<string, map<int, unsigned long>> contextWordFid;

	//map<string, int> featureCount; //used in constructing features 
	//unordered_map<string, unsigned long> featureStr2Id;
	//vector<string> features;	
	double *fvs;

	//FeatureIndexer(string featureFile);
	FeatureIndexer(string trainFile, Vocab *vocab, int direct_order);
	~FeatureIndexer(void) {
		if(fvs != NULL) {
			delete fvs;
		}
	}

	//void addFea(string featureStr);
	//long long getFeaId(string featureStr, bool add);
	//void toString(string outFile);
	void getFeatures(int* history, int direct_order, vector<int> &labels, vector<unsigned long> &featureIds);

private:
	void indexFeatures(vector<int> &history, int woc, int direct_order);	
	//void saveFeatureCount(string outFile, int minThreshold);

	
};

