#ifndef ARGUMENT_PARSER_HPP
#define ARGUMENT_PARSER_HPP

#include "globals.hpp"


class ArgumentParser
{
public:
	ArgumentParser(int argc, char **argv, bool canHaveSource, bool canHaveItrs);
	bool Parse();
	string GenerateHelpString();
	string GetDeviceName();

public:
	int argc;
	char** argv;
	
	bool canHaveSource;
	bool canHaveItrs;
	bool hasInput;
	bool hasSourceNode;
	bool hasOutput;
	bool hasDeviceID;
	bool hasNumberOfItrs;

	string input;
	int sourceNode;
	string output;	// the output file path and name
	int deviceID; // the GPU executing kernels
	int numberOfItrs;

	// parameters of setting evolving graphs
	int numSnapshots;
	double delta_rate_add;
	double delta_rate_del;
	double init_percentage;
	string schedule;
	uint degree_limit;
	uint bridge_thresh;
	uint low_level_thresh;
	uint degree_limit_DT;
};


#endif	//	ARGUMENT_PARSER_HPP
