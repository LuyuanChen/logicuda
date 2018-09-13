#pragma once

#ifndef KERNEL_H_
#define KERNEL_H_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"
#include <cublas.h>
#include <cublas_v2.h>
#include <cstdio>
#include "tensor.h"

#ifndef USE_TENSOR_CORES
#if __CUDACC_VER_MAJOR__ > 8
#define USE_TENSOR_CORES 1
#else
#define USE_TENSOR_CORES 0
#endif
#endif

#define BATCH_SIZE 200

template <typename T1, typename T2>
int matmul(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, cublasHandle_t cublas_handle);

void foo(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
#endif
