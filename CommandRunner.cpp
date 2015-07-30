#include "CommandRunner.h"
#include <string>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <unistd.h>

void CommandRunner::exec(const char* cmd) {
	static int MAX_ATTEMPTS = 10;
	int attempts = 0;
	FILE* pipe = NULL;	
	char buffer[128];
	std::stringstream sstm;
	
	while(true) {
		pipe = popen(cmd, "r");
		if (!pipe) {
			std::cerr << "error" << std::endl;
			exit(1);
		}
		attempts++;		
		//std::cerr << "attempt " << attempts << ": " << cmd << std::endl;
		sstm.str("");
		while(!feof(pipe)) {
			if(fgets(buffer, 128, pipe) != NULL) {
				sstm << buffer;
			}
		}
		pclose(pipe);
		std::string result = sstm.str();
		//std::cerr << result;
		//Job has been submitted. ID: 2730398. No error
		if (result.length() != 0) {
			break;
		} else {
			sleep(2000);
		}
		if(attempts == MAX_ATTEMPTS) {
			std::cerr << MAX_ATTEMPTS << " attempts reached!" << std::endl;
			std::cerr << cmd << std::endl;
			break;
		}
	}	
}

//int main(int argc, char **argv)
//{
//	//char* str = "job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM dir";
//	//char* str = "lsdf 2>&1";
//	char* str = "dir 2>&1";
//	CommandRunner::exec(str);
//}
