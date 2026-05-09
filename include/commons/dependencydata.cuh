#ifndef	DEPENDENCYDATA_CUH
#define	DEPENDENCYDATA_CUH

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include "globals.hpp"

struct DependencyData {
	uint parent;
	uint16_t level;
	uint16_t value;
	CUDA_HOSTDEV DependencyData() : level(DIST_INFINITY), value(), parent(PARENT_INFINITY) {}

	CUDA_HOSTDEV DependencyData(uint16_t _level, uint16_t _value, uint _parent)
		: level(_level), value(_value), parent(_parent) {}

	CUDA_HOSTDEV DependencyData(const DependencyData &object)
		: level(object.level), value(object.value), parent(object.parent) {}

	CUDA_HOSTDEV void reset() {
		parent = PARENT_INFINITY;
		level = DIST_INFINITY;
	}

	inline bool operator==(const DependencyData &rhs) {
		if ((value == rhs.value) && (parent == rhs.parent) &&
			(level == rhs.level))
			return true;
		else
			return false;
	}

	inline bool operator!=(const DependencyData &rhs) {
		if ((value != rhs.value) || (parent != rhs.parent) ||
			(level != rhs.level))
			return true;
		else
			return false;
	}

};

#endif 	//	DEPENDENCYDATA_HPP