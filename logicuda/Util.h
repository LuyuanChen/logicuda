#pragma once
#ifndef UTIL_H_
#define UTIL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"
#include <cstdio>
#include <cstdlib>
#include "helper_cuda.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int cudaInit()
{
	int devID = gpuGetMaxGflopsDeviceId();
	cudaDeviceProp deviceProp;

	gpuErrchk(cudaGetDeviceProperties(&deviceProp, devID));

	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

	return devID;
}

#endif