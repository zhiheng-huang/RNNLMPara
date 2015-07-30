#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdlib.h>

using namespace std;

class Word
{        
public:
	string word;
	int cn;
	//double prob;
	int classId;
	int superClassId;

	Word(string w, int c, double p, int cl, int scl)
	{
		word = w;
		cn = c;
		//prob = p;
		classId = cl;
		superClassId = scl;
	}

	//copy constructor
	Word(const Word &w) {
		word = w.word;
		cn = w.cn;
		//prob = w.prob;
		classId = w.classId;
		superClassId = w.superClassId;
	}
};

class Vocab
{
public:
	vector<Word> id2Word;
	map<string, int> word2Id;
	int wordCount;

	int classSize; //number of classes	
	int **classWords; //class id, word id per class -> word index in vocab (and in output layer)
	int *classCount; //word count for a class

	int superClassSize; //number of super classes
	int **superClassElements; //super class id, class id per super class -> class index
	int *superClassCount; //class count for a super class

	Vocab(string train_file);
	Vocab(ifstream &reader);
	Vocab(Vocab &vocab); 

	~Vocab() {
		for (int i=0; i<classSize; i++) {
			free(classWords[i]);
		}
		free(classWords);
		free(classCount);
	}

	int addWord(string word, int num);
	int getWordId(string word);
	string getWordStr(int wordId);
	int vocabSize();
	int getClass(int wordId);
	int numWordsInSameClass(int wordId);
	int numWordsInClass(int classId);
	int getWordIdFromClass(int classId, int wordIdInClass);
	int getWordCount();

	int getSuperClass(int wordId);
	int getClassIdFromSuperClass(int superClassId, int classIdInSuperClass);
	int numClassesInSameSuperClass(int wordId);

	void sortVocabByClass(string classFile);

	void sortVocabByFreq(int classNum, bool oldClass, int superClassNum, bool superClassEven);
	void syncWord2Id();

	void toString(string outFile);
	void toString(ofstream &writer);

protected:
	void assignClass(string classFile);
	void sortByClass();
	void buildClassWordIndex();

};

