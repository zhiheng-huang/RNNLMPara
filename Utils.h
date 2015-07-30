#pragma once
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class Utils
{
public:

    static double random(double min, double max)
    {
        return rand()/(double)RAND_MAX*(max-min)+min;
    }

    static void trim(string& str)
    {
        string::size_type pos1 = str.find_first_not_of(' ');
        string::size_type pos2 = str.find_last_not_of(' ');
        str = str.substr(pos1 == string::npos ? 0 : pos1, 
            pos2 == string::npos ? str.length() - 1 : pos2 - pos1 + 1);
    }

    static void lowercase(string &str)
    {
        transform(str.begin(), str.end(), str.begin(), ::tolower);
    }

    /*static void goToDelimiter(int delim, FILE *fi) {
    int ch=0;
    while (ch!=delim) {
    ch=fgetc(fi);
    if (feof(fi)) {
    printf("Unexpected end of file\n");
    exit(1);
    }
    }
    }*/

    static string bool2Str(bool b) {
        if(b) {
            return "true";
        } else {
            return "false";
        }
    }

    //Test if a given file contains a string or not
    static bool ContainsStr(string fileName, string str)
    {
        ifstream reader(fileName);
        string line;
        bool found = false;
        while (getline(reader, line))
        {
            trim(line);
            if (line.compare(str) == 0)
            {
                found = true;
                break;
            }
        }
        reader.close();
        return found;
    }

    static string getVal(ifstream &reader)
    {
        string line;
        while (getline(reader, line))
        {
            size_t pos = line.find(":");
            if (pos != string::npos)
            {
                string res = line.substr(pos + 1); //empty string if no string
                trim(res);
                //if (res.Length == 0) throw new Exception("error");
                return res;
            }
        }
        return "";
    }

    static bool str2Bool(std::string str) {
        Utils::lowercase(str);
        Utils::trim(str);

        if(str.compare("true")==0) {
            return true;
        } else {
            return false;
        }
    }

    static double str2Double(string str) {
        std::stringstream i(str);
        double x;
        if (!(i >> x))
            return 0;
        return x;
    }

    static void Tokenize(const string& str,
        vector<string>& tokens,
        const string& delimiters = " ")
    {
        // Skip delimiters at beginning.
        string::size_type lastPos = str.find_first_not_of(delimiters, 0);
        // Find first "non-delimiter".
        string::size_type pos     = str.find_first_of(delimiters, lastPos);

        while (string::npos != pos || string::npos != lastPos)
        {
            // Found a token, add it to the vector.
            tokens.push_back(str.substr(lastPos, pos - lastPos));
            // Skip delimiters.  Note the "not_of"
            lastPos = str.find_first_not_of(delimiters, pos);
            // Find next "non-delimiter"
            pos = str.find_first_of(delimiters, lastPos);
        }
    }

    static void vector2Str(vector<int> &strs, string delimiter, string &output) {
        stringstream ss;
        for(int i = 0; i < strs.size(); i++) {
            ss << strs[i]; 
            if(i != strs.size()-1) {
                ss << delimiter;
            }
        }
        output = ss.str();
    }

    static void getLogPs(string rnnlm_file, vector<double>& res)
    {		
        ifstream fi(rnnlm_file, ios::in | ios::binary);	
        if(!fi.good()) {
            cerr << "model file " << rnnlm_file << " is not existing" << endl;
            exit(1);
        }
        float fl;
        int a, b;
        string line;

        int version = atoi(Utils::getVal(fi).c_str());
        int rand_seed = atoi(Utils::getVal(fi).c_str());
        //srand(rand_seed);
        string s = Utils::getVal(fi);
        vector<int> context_ids;
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
        bool file_binary = Utils::str2Bool(Utils::getVal(fi));
        string train_file = Utils::getVal(fi);
        string valid_file = Utils::getVal(fi);
        double llogpValid = Utils::str2Double(Utils::getVal(fi));
        double logpValid = Utils::str2Double(Utils::getVal(fi));
        fi.close();
        res.clear();
        res.push_back(llogpValid);
        res.push_back(logpValid);
    }

};
