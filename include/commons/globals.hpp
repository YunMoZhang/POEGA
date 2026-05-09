#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <string>
#include <ctime>
#include <random>
#include <stdio.h>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <sstream> 

using namespace std;

const unsigned int DIST_INFINITY = std::numeric_limits<uint16_t>::max()/2;
const unsigned int PARENT_INFINITY = std::numeric_limits<uint>::max()/2;

typedef unsigned int uint;
typedef unsigned long long ull;

struct OutEdge{
    uint end;
};

struct OutEdgeWeighted{
    uint end;
    uint w8;
};

struct Edge{
	uint source;
    uint end;
};

struct EdgeWeighted{
	uint source;
    uint end;
    uint w8;
};


struct OutEdge_Evolving{
    uint end;
    ull bitmap; 
};

struct OutEdgeWeighted_Evolving{
    uint end;
    uint w8;
    ull bitmap;
};

struct Edge_Evolving{
	uint source;
    uint end;
    ull bitmap;
};

struct Edge_Union{
	uint source;
    uint end;
    int snap_a;
    int snap_d;
};

struct EdgeWeighted_Evolving{
	uint source;
    uint end;
    uint w8;
    ull bitmap;
};

struct EdgeWeighted_Union{
	uint source;
    uint end;
    uint w8;
    int snap_a;
    int snap_d;
};

#endif 	//	GLOBALS_HPP
