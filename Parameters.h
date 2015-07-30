#pragma once
#include <map>
#include <string>

using namespace std;

class Parameters
{
public:
	map<string, string> parameters;

	Parameters(string file);
	string getPara(string key);
	bool getParaBool(string key);
	int getParaInt(string key, int def);
	//long long getParaLong(string &key);
	double getParaDouble(string key, double def);
};

