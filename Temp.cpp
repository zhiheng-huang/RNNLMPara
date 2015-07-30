#include "Temp.h"
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include "RNN.h"

using namespace std;

Temp::Temp(void)
{
}


Temp::~Temp(void)
{
}

//int main(int argc, char **argv)
//{
	//	ofstream myFile ("//fbl/nas/HOME/zhihuang/RNNParaExp/Temp", ios::out | ios::binary);
	//	myFile << "this is text" << endl;
	//	myFile.write("t", 1);
	//	float fl = 0.0456789;
	//	myFile.write((char*)&fl, sizeof(float));
	//	myFile.close();
	//
	//	ifstream myFile2 ("//fbl/nas/HOME/zhihuang/RNNParaExp/Temp", ios::in | ios::binary);
	//	string line;
	//	char c;
	//	float fl2;
	//	//float fl;
	//	getline(myFile2, line); 
	//	cout << line << endl;
	//	myFile2 >> c;
	//	cout << c << endl;
	//	myFile2.read((char*)&fl2, sizeof(float));
	//	std::cout.precision(4);
	//	cout << fl2 << endl;
	//	myFile2.close();
	//	
	//
	//string modelFile = "//fbl/nas/HOME/zhihuang/RNNParaExp/ispeech20MBatch/modelRnnIter0";
	//string modelFile2 = "//fbl/nas/HOME/zhihuang/RNNParaExp/ispeech20MBatch/modelRnnIter0-2";
	//string modelFile3 = "//fbl/nas/HOME/zhihuang/RNNParaExp/ispeech20MBatch/modelRnnIter0-3";
	//RNN model(modelFile);
	//RNN model3(model);
	//model3.alpha = 10;
	//model.saveNet(modelFile2);
	//model3.saveNet(modelFile3);

	//for (int i = 0; i < 5; i++)
	//{
	//	std::stringstream sstm;
	//	string rnnModelFile = "//fbl/nas/HOME/zhihuang/RNNParaExp/ispeech10MBatch/modelRnn";
	//	RNN* model = new RNN(rnnModelFile, true); //still cannot set false, as the net will need vocab size info
	//	cout << "***** Start loading RNN model" << i << endl;
	//	//may add criteria here to update good models only!
	//	delete model;
	//	model = NULL;
	//}

//}