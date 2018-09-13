#pragma once
#ifndef CUDA_HELPER_H_
#define CUDA_HELPER_H_

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		// Make sure we call CUDA Device Reset before exiting
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

#endif // !CUDA_HELPER_H_

