#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"
#include "numpy.hpp"
#include "Util.h"
#include "kernel.h"
#include "helper_cuda.h"
#include <cublas_v2.h>


void load_file(const std::string filename);

int main()
{
	std::cout << "Hello World!\n";

	cudaInit();
	cublasHandle_t cublas_handle;
	cudnnHandle_t cudnn_handle;
	checkCublasErrors(cublasCreate(&cublas_handle));
	cudnnCreate(&cudnn_handle);
	foo(cublas_handle, cudnn_handle);

	std::cin.get();
	return 0;
}

void load_file(const std::string filename) {
	std::vector<int> shape;
	int h, w, l;
	std::vector<int> data;

	aoba::LoadArrayFromNumpy(filename, h, w, l, data);
	std::cout << shape[0] << " " << shape[1] << std::endl; // 4 5
	std::cout << data.size() << std::endl; // 20
}

