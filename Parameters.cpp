#include "Parameters.h"
#include <fstream>
#include <sstream>
#include <assert.h>
#include <stdlib.h>
#include "Utils.h"

Parameters::Parameters(string file)
{
	ifstream reader(file);
	string line;
	while(getline(reader, line)) {
		if(line.empty() || line.find("#") == 0) {
			continue;
		}
		size_t index = line.find("=");
		if(index != string::npos) {
			string key = line.substr(0, index);
			string val = line.substr(index+1);
			size_t p = val.find("#");
			if(p!= string::npos && p > 0) {
				val = val.substr(0, p);
			}
			Utils::trim(key);
			Utils::trim(val);
			assert(!key.empty());
			assert(!val.empty());
			parameters.insert(pair<string, string>(key, val));
		}
	}
	reader.close();
}


string Parameters::getPara(string key)
{
	if( parameters.find(key) == parameters.end()) {
		return "";
	}	
	return parameters[key];
}

bool Parameters::getParaBool(string key)
{
	string val = getPara(key);
	return Utils::str2Bool(val);
}

int Parameters::getParaInt(string key, int def)
{
	string s = getPara(key);
	if(s.empty()) {
		return def;
	} else {
		return atoi(s.c_str());
	}
}

double Parameters::getParaDouble(string key, double def)
{
	string val = getPara(key);
	if(val.empty()) {
		return def;
	} else {
		return Utils::str2Double(val);
	}
}


