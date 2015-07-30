#include "FeatureIndexer.h"
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include "Utils.h"
#include <assert.h>


//FeatureIndexer::FeatureIndexer(string featureFile)
//{
//	featureSize = 0;
//	ifstream infile(featureFile);
//	string line;
//	while (getline(infile, line)) {
//		Utils::trim(line);
//		if(line.empty()) continue;
//		vector<string> comps;
//		Utils::Tokenize(line, comps, "\t");
//		if(comps[0].empty()) continue;
//		addFea(comps[0]);
//	}
//	infile.close();
//	fvs = new double[features.size()];
//	for(int i = 0; i < features.size(); i++) {
//		fvs[i] = Utils::random(-0.1, 0.1);
//	}
//}

FeatureIndexer::FeatureIndexer(string trainFile, Vocab *vocab, int direct_order) {
	featureSize = 0;
	fvs = NULL;
	ifstream is(trainFile);
	string line;
	while(getline(is, line)) {
		if(line.empty()) continue;
		vector<string> tokens;
		Utils::Tokenize(line, tokens, " ");
		tokens.push_back("</s>");
		vector<int> history;		
		for(int i = 0; i < tokens.size(); i++) {
			int wordId = vocab->getWordId(tokens[i]);
			assert(wordId != -1);
			indexFeatures(history, wordId, direct_order);
			int wordClass = vocab->getClass(wordId);
			assert(wordClass != -1);
			indexFeatures(history, vocab->vocabSize() + wordClass, direct_order);
			history.insert(history.begin(), wordId);
		}
	}
	is.close();

	//for(map<string, int>::iterator iter = featureCount.begin(); iter != featureCount.end(); iter++) {
	//	if(iter->second >= minCount) {
	//		addFea(iter->first);
	//	}
	//}
	//featureCount.clear(); //release memory
	fvs = new double[featureSize];
	for(int i = 0; i < featureSize; i++) {
		fvs[i] = Utils::random(-0.1, 0.1);
	}
}


//void FeatureIndexer::addFea(string featureStr) {
//	if(featureStr2Id.find(featureStr) != featureStr2Id.end()) {
//		std::cerr << "duplicated feature " << featureStr << endl;
//		exit(1);
//	}
//	long long size = featureStr2Id.size();
//	featureStr2Id.insert(pair<string, unsigned long>(featureStr, size));
//	features.push_back(featureStr);
//}

void FeatureIndexer::indexFeatures(vector<int> &history, int woc, int direct_order) {
	std::stringstream ss;
	//probably remove i = 0 case, which reduces computation
	for (int i = 0; i < direct_order; i++) {
		if(i == 1) {
			if(i-1 < history.size()) {
				ss << history[i-1];
			} else {
				ss << 0;
			}
		} else if (i > 1) {
			if(i-1 < history.size()) {
				ss << "-" << history[i-1];
			} else {
				ss << "-" << 0;
			}
		}		
		string f = ss.str();
		if(contextLabelFId.find(f) == contextLabelFId.end()){
			vector<pair<int, unsigned long> > v;
			contextLabelFId[f] = v;
		} 

		//map<int, unsigned long> m = contextLabelFId[f];
		bool found = false;
		for(int i = 0; i < contextLabelFId[f].size(); i++) {
			if(contextLabelFId[f].at(i).first == woc) {
				found = true;
				break;
			}
		}
		if(found == false) {		
			contextLabelFId[f].push_back(pair<int, unsigned long>(woc,featureSize++));
		}
	}
}

//void FeatureIndexer::toString(string outFile)
//{
//	ofstream writer(outFile);
//	for(map<string, vector<pair<int, unsigned long>>>::iterator iter = contextLabelFId.begin(); iter != contextLabelFId.end(); iter++) {
//		writer << iter->first ;
//		for(map<int, unsigned long>::iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {
//			writer << ":" << iter2->first << "=" << iter2->second;
//		}
//		writer << endl;
//	}
//	writer.close();
//}

void FeatureIndexer::getFeatures(int* history, int direct_order, vector<int> &labels, vector<unsigned long> &featureIds) {
	std::stringstream ss;
	//probably remove i = 0 case, which reduces computation
	for (int i = 0; i < direct_order; i++) {
		if(i == 1) {
				ss << history[i-1];			
		} else if (i > 1) {
				ss << "-" << history[i-1];			
		}		
		string f = ss.str();
		if(contextLabelFId.find(f) != contextLabelFId.end()){
			for(vector<pair<int, unsigned long> >::iterator iter = contextLabelFId[f].begin(); iter != contextLabelFId[f].end(); iter++) {
				labels.push_back(iter->first);
				featureIds.push_back(iter->second);
			}
		} 
	}
}



//void FeatureIndexer::saveFeatureCount(string outFile, int minThreshold) {
//	ofstream writer(outFile);
//	for(map<string, int>::iterator iter = featureCount.begin(); iter != featureCount.end(); iter++) {
//		if(iter->second >= minThreshold) {
//			writer << iter->first << "\t" << iter->second << endl;
//		}
//	}
//	writer.close();
//}


int main2(int argc, char **argv)
{
	/*FeatureIndexer fi;
	fi.addFea("a1|b1");
	fi.addFea("bafdsaf|fdsa");
	string file = "c:/temp/featureFile";
	string file2 = "c:/temp/featureFile2";
	fi.toString(file);
	long long id = fi.getFeaId("bafdsaf|fdsa", false);
	id = fi.getFeaId("aa", false);
	id = fi.getFeaId("aa", true);
	FeatureIndexer fi2(file);
	fi2.toString(file2);
	return 0;*/
	return 0;
}
