#include "Vocab.h"
#include "Utils.h"
#include <fstream>
#include <sstream>
#include <set>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>

//Constructor from a training file.
Vocab::Vocab(string train_file)
{
	wordCount = 0;
	addWord("</s>", 0);
	ifstream infile(train_file.c_str());
	string line;
	while (getline(infile, line)) {
		Utils::trim(line);
		if(line.length() == 0) continue;
		vector<string> event;
		vector<string> tokens;
		Utils::Tokenize(line, event, "\t");
		if(event.size() == 0) continue;
		Utils::Tokenize(event[0], tokens, " ");
		if(tokens.size() == 0) continue;
		for (int i = 0; i < tokens.size(); i++) {
			string token = tokens[i];
			addWord(token, 1);
			wordCount++;
		}
		addWord("</s>", 1);
		wordCount++;			
	}
	classSize = -1;
	superClassSize = -1;
}

//Constructor from a saved Vocab object. Words in the same class 
//should be continuously located. Classes are sorted in ascending
//order (0-based). classSize and superClasses are initialized.
Vocab::Vocab(ifstream &reader)
{
	string line;
	string token;

	wordCount = 0;
	while (getline(reader, line)) {
		if(line.compare("Vocabulary:") == 0) {
			break;
		}
	}
	set<int> classes;
	set<int> superClasses;
	while(getline(reader, line)) {
		if(line.empty()) break;
		stringstream ss(line);
		vector<string> tokens;
		while (ss >> token) {
			tokens.push_back(token);
		}
		string ws = tokens[2];
		int count = atoi(tokens[1].c_str());
		wordCount += count;
		int superClassId = -1; //backward compatible
		if(tokens.size() >=5) {
			superClassId = atoi(tokens[4].c_str());
		}
		Word w(ws, count, 0, atoi(tokens[3].c_str()), superClassId);
		int wordId = atoi(tokens[0].c_str());
		id2Word.push_back(w);
		word2Id.insert(pair<string, int>(w.word, wordId));
		classes.insert(w.classId);
		superClasses.insert(w.superClassId);
	}
	classSize = classes.size();
	superClassSize = superClasses.size();
	buildClassWordIndex();
}

//copy constructor //TODO: super classes
Vocab::Vocab(Vocab &vocab) {	
	for(int i = 0; i < vocab.id2Word.size(); i++) {
		Word w(vocab.id2Word[i]);
		id2Word.push_back(w);
	}
	for(map<string, int>::iterator iter = vocab.word2Id.begin(); iter != vocab.word2Id.end(); iter++) {
		word2Id[iter->first] = iter->second;
	}
	wordCount = vocab.wordCount;
	classSize = vocab.classSize;

	classWords=(int **)calloc(classSize, sizeof(int *));
	classCount=(int *)calloc(classSize, sizeof(int));
	for (int i = 0; i < classSize; i++)
	{
		classCount[i] = vocab.classCount[i];
		classWords[i]=(int *)calloc(classCount[i], sizeof(int));
		for(int j = 0; j < classCount[i]; j++) {
			classWords[i][j] = vocab.classWords[i][j];
		}		
	}
	//TODO: super classes
	superClassSize = -1;
}

//add word to word2Id and update id2Word, return word index
int Vocab::addWord(string word, int num)
{
	int wordId = -1;
	if(word2Id.find(word) == word2Id.end()) {
		Word w(word, 0, 0, -1, -1);
		wordId = word2Id.size();
		word2Id.insert(pair<string, int>(word, wordId));
		id2Word.push_back(w);
	} else {
		wordId = word2Id[word];
	}
	id2Word[wordId].cn += num;
	return wordId;
}

//Return word id for a given word. Return -1 if it is not in vocabulary
int Vocab::getWordId(string word)
{
	if (word2Id.find(word) != word2Id.end())
	{
		return word2Id[word];
	}
	else
	{
		return -1;
	}
}

string Vocab::getWordStr(int wordId)
{
	return id2Word[wordId].word;
}

int Vocab::vocabSize()
{
	return id2Word.size();
}

int Vocab::getClass(int wordId)
{
	return id2Word[wordId].classId;
}

int Vocab::numWordsInSameClass(int wordId)
{
	return classCount[getClass(wordId)];
}

int Vocab::numWordsInClass(int classId)
{
	return classCount[classId];
}

int Vocab::getWordIdFromClass(int classId, int wordIdInClass)
{
	return classWords[classId][wordIdInClass];
}

int Vocab::getWordCount()
{
	return wordCount;
}

int Vocab::getSuperClass(int wordId)
{
	return id2Word[wordId].superClassId;
}

int Vocab::getClassIdFromSuperClass(int superClassId, int classIdInSuperClass)
{
	return superClassElements[superClassId][classIdInSuperClass];
}

int Vocab::numClassesInSameSuperClass(int wordId)
{
	return superClassCount[getSuperClass(wordId)];
}

//////////////////////////////////////////////
////      class related operations        ////
//////////////////////////////////////////////

//sort vocabulary so that words belong to the same class are located continously
void Vocab::sortVocabByClass(string classFile)
{
	assignClass(classFile);         
	sortByClass();
	buildClassWordIndex();
	syncWord2Id();
}

//read class file and assign classes to words in id2Word. </s> must be in class
//file but <s> must not. </s> is re-assigned to the maximum class number
void Vocab::assignClass(string classFile)
{
	ifstream reader(classFile);
	map<string, int> word2Class;
	set<int> classes;
	string line;
	string token;
	int endTagClass = -1;
	int maxClass = -1;
	while(getline(reader, line)) {
		vector<string> tokens;
		stringstream ss(line);
		while(ss >> token) {
			tokens.push_back(token);
		}
		assert(tokens.size() == 2);
		assert(tokens[0].compare("<s>") != 0);
		string w = tokens[0];
		int classId = atoi(tokens[1].c_str());
		word2Class.insert(pair<string, int>(w, classId));
		classes.insert(classId);
		maxClass = (classId > maxClass) ? (classId) : (maxClass);
		if (w.compare("</s>") == 0)
		{
			endTagClass = classId;
		}
	}          
	classSize = classes.size();
	reader.close();

	if (endTagClass == -1)
	{
		cerr << "</s> must be present in the vocabulary\n";
		exit(1);
	}           

	// </s> needs to have the highest class index because it needs to come first in the vocabulary...
	//first sort words in descending order of their classes (</s> in the first position). then
	//change the class numbers from 0 (</s> still in the first postion)
	for(map<string, int>::iterator it = word2Class.begin(); it != word2Class.end(); it++) {
		if (it->second == endTagClass)
		{
			it->second = maxClass;
		}
		else if (it->second == maxClass)
		{
			it->second = endTagClass;
		}
	}
	assert(word2Class.size() == id2Word.size());
	for (int i = 0; i < id2Word.size(); i++)
	{
		string w = id2Word[i].word;
		if(word2Class.find(w) == word2Class.end())
		{
			cerr << w << " missing from class file\n";
			exit(1);
		}
		id2Word[i].classId = word2Class[w];
	}
}

//sort vocab in ascending order of class index. After this call, vocab_hash is invalid?
void Vocab::sortByClass()
{
	int max;
	for (int i = 1; i < id2Word.size(); i++)
	{
		max = i;
		for (int j = i + 1; j < id2Word.size(); j++)
		{
			if (id2Word[max].classId < id2Word[j].classId)
			{
				max = j;
			}
		}
		Word swap = id2Word[max];
		id2Word[max] = id2Word[i];
		id2Word[i] = swap;
	}

	int cnum = -1, last = -1;
	//word was previously sorted in descending order of class index
	//now change class index to ascending order
	for (int i = 0; i < id2Word.size(); i++)
	{
		if (id2Word[i].classId != last)
		{
			last = id2Word[i].classId;
			id2Word[i].classId = ++cnum;
		}
		else
		{
			id2Word[i].classId = cnum;
		}
	}
}

//sort vocab in descending order of counts. After this call, vocab_hash is invalid?
void Vocab::sortVocabByFreq(int classNum, bool oldClass, int superClassNum, bool superClassEven)
{
	int max;
	classSize = classNum;
	superClassSize = superClassNum;
	//why ignore 0th element? because it is </s>?
	for (int i = 1; i < id2Word.size(); i++)
	{
		max = i;
		for (int j = i + 1; j < id2Word.size(); j++)
		{
			if (id2Word[max].cn < id2Word[j].cn)
			{
				max = j;
			}
		}
		Word swap = id2Word[max];
		id2Word[max] = id2Word[i];
		id2Word[i] = swap;
	}

	int index = 0;
	int count = 0;
	double df = 0;
	double dd = 0;

	if (oldClass)
	{ // old classes
		//assign word class in descending order of word count
		for (int i = 0; i < id2Word.size(); i++)
		{
			count += id2Word[i].cn;
		}
		for (int i = 0; i < id2Word.size(); i++)
		{
			df += id2Word[i].cn / (double)count;
			if (df > 1)
			{
				df = 1;
			}
			if (df > (index + 1) / (double)classSize)
			{
				id2Word[i].classId = index;
				if (index < classSize - 1)
				{
					index++;
				}
			}
			else
			{
				id2Word[i].classId = index;
			}
		}
	}
	else
	{
		int* classWordCount = (int *)calloc(classSize, sizeof(int));
		for(int i = 0; i < classSize; i++) {
			classWordCount[i] = 0;
		}
		// povey-style new classes
		for (int i = 0; i < id2Word.size(); i++)
		{
			count += id2Word[i].cn;
		}
		for (int i = 0; i < id2Word.size(); i++)
		{
			dd += sqrt(id2Word[i].cn / (double)count);
		}
		for (int i = 0; i < id2Word.size(); i++)
		{
			df += sqrt(id2Word[i].cn / (double)count) / dd;
			if (df > 1) df = 1;
			id2Word[i].classId = index;
			classWordCount[index] += id2Word[i].cn;
			if (df > (index + 1) / (double)classSize)
			{				
				if (index < classSize - 1)
				{
					index++;
				}
			}				
		}

		//super class
		if(superClassSize > 1) {			
			if(superClassEven == true) {
				int num = classSize/superClassSize;
				int n = classSize - superClassSize*num; //number of super classes which have num+1 classes
				for (int i = 0; i < id2Word.size(); i++)
				{
					int classId = id2Word[i].classId;
					if(classId < n*(num+1)) {
						id2Word[i].superClassId = classId/(num+1);
					} else {
						id2Word[i].superClassId = (classId-n*(num+1))/num + n;
					}
				}
			} else {
				dd = 0;
				for (int i = 0; i < classSize; i++)
				{
					dd += sqrt(classWordCount[i] / (double)count);
				}
				df = 0;
				index = 0;
				int prevClassId = -1;
				for (int i = 0; i < id2Word.size(); i++)
				{
					int classId = id2Word[i].classId;
					if(classId != prevClassId) {
						df += sqrt(classWordCount[classId] / (double)count) / dd;
					}
					if (df > 1) df = 1;				
					if (df > (index + 1) / (double)superClassSize)
					{				
						if (index < superClassSize - 1)
						{
							index++;
						}
					}		
					id2Word[i].superClassId = index;
					prevClassId = classId; 
				}
			}
		}
	}
	buildClassWordIndex();
	syncWord2Id(); //have to re-sync!!
}

//Build class word index by populating classCount, classWords, superClassCount,
//and superClassElements. Prior to this call, words in the same class should be 
//continuously located. Classes are sorted in ascending order (0-based). 
void Vocab::buildClassWordIndex()
{
	cout << "classSize: " << classSize << " superClassSize:" << superClassSize << endl;
	vector<vector<int> > tmp;
	for (int i = 0; i < classSize; i++)
	{
		tmp.push_back(vector<int>());
	}
	for (int i = 0; i < id2Word.size(); i++)
	{
		int cl = id2Word[i].classId;
		tmp[cl].push_back(i);
	}

	//classWords = new int*[classSize];
	//classCount = new int[classSize];
	classWords=(int **)calloc(classSize, sizeof(int *));
	classCount=(int *)calloc(classSize, sizeof(int));
	for (int i = 0; i < classSize; i++)
	{
		if (tmp[i].size() == 0)
		{
			cerr << "ERROR: class " << i << " has no members\n";
			exit(1);
		}
		//classWords[i] = new int(tmp[i].size());
		classWords[i]=(int *)calloc(tmp[i].size(), sizeof(int));
		for(int j = 0; j < tmp[i].size(); j++) {
			classWords[i][j] = tmp[i][j];
		}
		classCount[i] = tmp[i].size();
	}

	if(superClassSize > 1) {
		tmp.clear();
		for (int i = 0; i < superClassSize; i++)
		{
			tmp.push_back(vector<int>());
		}
		for (int i = 0; i < id2Word.size(); i++)
		{
			int sc = id2Word[i].superClassId;
			int cl = id2Word[i].classId;
			if(std::find(tmp[sc].begin(), tmp[sc].end(), cl) == tmp[sc].end()) {
				tmp[sc].push_back(cl);
			}
		}

		superClassElements=(int **)calloc(superClassSize, sizeof(int *));
		superClassCount=(int *)calloc(superClassSize, sizeof(int));
		for (int i = 0; i < superClassSize; i++)
		{
			if (tmp[i].size() == 0)
			{
				cerr << "ERROR: super class " << i << " has no members\n";
				exit(1);
			}
			superClassElements[i]=(int *)calloc(tmp[i].size(), sizeof(int));
			for(int j = 0; j < tmp[i].size(); j++) {
				superClassElements[i][j] = tmp[i][j];
				//cerr << "superClassElements["<< i << "][" << j <<"]= " << superClassElements[i][j]<< endl;
			}
			superClassCount[i] = tmp[i].size();
			//cerr << "superClassCount[" << i << "] = " << tmp[i].size() << endl;
		}
	}
}

//Re-index word to id mapping due to swaps of words in vocab
void Vocab::syncWord2Id()
{
	word2Id.clear();
	for (int i = 0; i < id2Word.size(); i++)
	{
		string ws = id2Word[i].word;
		word2Id.insert(pair<string, int>(ws, i));
	}
}

void Vocab::toString(string outFile)
{
	ofstream writer(outFile);
	writer << "Vocabulary:" << endl;
	toString(writer);
	writer.close();
}

void Vocab::toString(ofstream &writer)
{
	for (int i = 0; i < id2Word.size(); i++)
	{
		Word word = id2Word[i];
		writer << i << "\t" << word.cn << "\t" << word.word << "\t" << word.classId << endl;
		//writer << i << "\t" << word.cn << "\t" << word.word << "\t" << word.classId << "\t" << word.superClassId << endl;
	}
}

int main3(int argc, char **argv)
{
	string trainFile = "//fbl/nas/home/zhihuang/RNNOrigExp/small/train";
	string vocabFile = "//fbl/nas/home/zhihuang/RNNOrigExp/small/trainVocab";
	string vocabFile2 = "//fbl/nas/home/zhihuang/RNNOrigExp/small/trainVocab2";
	string classFile = "//fbl/nas/home/zhihuang/RNNOrigExp/small/classFile";
	int classNum = 9;
	bool oldClass = false;	
	int superClassNum = 3;
	bool superClassEven= true;
	Vocab vocab(trainFile);
	vocab.sortVocabByFreq(classNum, oldClass, superClassNum, superClassEven);
	//vocab.sortVocabByClass(classFile);
	vocab.toString(vocabFile);

	ifstream reader(vocabFile);
	Vocab vocab2(reader);
	vocab2.toString(vocabFile2);
	reader.close();
	return 0;
}
