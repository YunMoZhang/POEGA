#ifndef OUTPUT_UTILITIES_HPP
#define OUTPUT_UTILITIES_HPP

#include "globals.hpp"
#include "dependencydata.cuh"

namespace utilities {
	void PrintResults(uint *results, uint n);
	void PrintResults(float *results, uint n);
	void PrintResults(double *results, uint n);
	void PrintResults(DependencyData *results, uint n);
	void PrintParents(DependencyData *results, uint n);
	void SaveResults(string filepath, uint *results, uint n);
	void SaveResults(string filepath, float *results, uint n);
	void SaveResults(string filepath, double *results, uint n);
}

#endif	//	OUTPUT_UTILITIES_HPP