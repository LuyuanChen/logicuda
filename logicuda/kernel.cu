#include "kernel.h"
#include "helper_cuda.h"

template <typename T1, typename T2>
int matmul(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, cublasHandle_t cublas_handle)
{
	const float alpha = 1.f;
	const float beta = 1.f;

	int m = C.dims()[0];
	int k = A.dims()[1];
	int n = C.dims()[1];

	cudaDataType_t A_type = CUDA_R_32F;  // might be CUDA_R_32I/8I
	cudaDataType_t B_type = CUDA_R_32F;
	cudaDataType_t C_type = CUDA_R_32F;
	cudaDataType_t compute_type = CUDA_R_32F;
	cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
	cublasStatus_t stat;

	// cublas uses column major, meaning that devA = hostA(T). Such that, devA(T) * devB(T) = devAB(T) = hostAB.
	// this means, doing gemm(B,A) = AB in host memory. Note though, cublasGemmEx uses device memory
	// TODO investigate the transpose and A,B ordering
	//cublasGemmEx(cublas_handle,
	//	CUBLAS_OP_N,
	//	CUBLAS_OP_N,
	//	m,
	//	n,
	//	k,
	//	&alpha,
	//	A.begin(), A_type, A.dims()[0],
	//	B.begin(), B_type, B.dims()[0],
	//	&beta,
	//	C.begin(), C_type, C.dims()[0],
	//	compute_type,
	//	algo);

	cublasSgemm(cublas_handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		m,
		n,
		k,
		&alpha,
		A.begin(), A.dims()[0],
		B.begin(), B.dims()[0],
		&beta,
		C.begin(), C.dims()[0]);


	// TODO for adding a bias term, can use the beta as 1 and add b*I, where I is an identity matrix
	cudaDeviceSynchronize();
	return 0;
}


// logits = wx+b, only wx is available for now
template <typename T>
void sigmoid(Tensor<T> logits, Tensor<T> results, cudnnHandle_t cudnnHandle) 
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	int batch_size = 20;
	// a batch_size x 1 vector, out put should be the same shape, so no need to define again
	cudnnTensorDescriptor_t logits_desc;
	cudnnCreateTensorDescriptor(&logits_desc);
	cudnnSetTensor4dDescriptor(logits_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		logits.dims()[0], 1, 1, 1);

	cudnnActivationDescriptor_t sigmoid_desc;
	cudnnCreateActivationDescriptor(&sigmoid_desc);
	cudnnSetActivationDescriptor(sigmoid_desc,
		CUDNN_ACTIVATION_SIGMOID,
		CUDNN_PROPAGATE_NAN,
		0.0);

	cudnnActivationForward(cudnnHandle,
		sigmoid_desc,
		&alpha,
		logits_desc,  // input desc
		logits.begin(),
		&beta,
		logits_desc,  // output desc
		results.begin()
	);

	cudnnDestroyActivationDescriptor(sigmoid_desc);
	cudnnDestroyTensorDescriptor(logits_desc);
}


// __float2int_rd
template <typename T1, typename T2>
void loss_backprop(Tensor<T1>x, Tensor<T1> results, Tensor<T2> labels, cublasHandle_t cublasHandle)
{
	// the backprop for sigmoid loss is (h-y)x_i for each weight. So the matrix for gradient for N training
	// examples will be in size Nxd
	// h-y is, firstly, a Nx1 matrix, x is a Nxd matrix, each row of x needs to be multiplexed by the corresponding h-y scaler value
	// we operate here on results directly, to save space
	// TODO: problem that might occur: type T1 and T2 not directly compatible

	// firstly, calculate h-y, assume both are float for now, TODO: may require cast from int to float for labels. AXPY does alpha*x + y,
	// rewriting y, we want result to be rewriten, so essentially x -> labels, y -> results
	const float alpha = -1.0f; 
	checkCudaErrors(cublasSaxpy_v2(cublasHandle,
		results.dims[0],  // num of elem
		&alpha,
		labels.begin(), 1, // x, stride x
		results.begin(), 1 // y, stride y
	));

	// result is now h-y, now needs to multiply each row of it (a scaler) to x matrix, which is of size Nxd
	// this can be done by making h-y a diag matrix, diag(h-y) * x will be the answer, h-y -> X, x -> A
	checkCudaErrors(cublasSdgmm(cublasHandle,
		CUBLAS_SIDE_LEFT,
		x.dims[0],  // rows
		x.dims[1],  // cols
		x.begin(),  // A
		x.dims[0],  // lda
		labels.begin(),  // x
		1,			// incx
		x.begin(),  // C, store the result in x
		x.dims[0]));

	// now x stores the gradient values for each weight for each training example in the batch. Now, average them
}

template <typename T1, typename T2>
void update_weights(Tensor<T1>dw, Tensor<T1> weight, cublasHandle_t cublasHandle, float lr)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	// dw is of size Nxd, where N is batch size, first have to average it, by multiply a vector
	// make a nx1 vector that has values 1/n. A->dw, x->dw, y -> weights
	int n = dw.dims[0];
	int d = dw.dims[1];
	float n_reci = 1 / static_cast<float>(n);
	std::vector<int> dim = { n, 1 };
	Tensor<float> reduction_vec = fill<float>(dim, n_reci);
	Tensor<T1> average_weight({ d, 1 });

	// average_weights = average(dw, axis=row)
	cublasSgemv_v2(cublasHandle, CUBLAS_OP_T,
		n, d, // rows and cols in A
		&alpha,
		dw.begin(), dw.dims[0], // A
		reduction_vec.begin(), 1,
		&beta,
		average_weight.begin(), 1);

	lr *= -1;
	// update the weights, weights(Y) = -lr(A)*average weight (X) + weights (Y)
	cublasSaxpy_v2(cublasHandle,
		d,
		&lr,
		average_weight.begin(), 1,
		weight.begin(), 1);
}

// given feature vector, weight and calculate the coresponding probability
// the input should be in the device memory already. Suppose that wx is directly multiplicable.
// inference = h(w, x) = sigmoid(wx+b)
template <typename T1, typename T2>
void inference(Tensor<T1> w, Tensor<T1> x, Tensor<T2> result, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
{
	// first calculate w*x, result is a column vector strored in result
	matmul(w, x, result, cublasHandle);

	// inplace sigmoid, not sure if it's safe to do that
	sigmoid(result, result, cudnnHandle);
}

void foo(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
{
	{ printf("Hello cublas\n"); }

	printf("Creating tensor with size 5x5\n");
	std::vector<int> dim = { 5, 5 };
	Tensor<float> t1 = fill<float>(dim, 1.5);

	float h_f;
	cudaMemcpy(&h_f, t1.begin(), sizeof(float) , cudaMemcpyDeviceToHost);
	printf("Content of start: %f\n", h_f);
	std::cout << t1 << std::endl;

	Tensor<float> t2 = fill<float>({ 5, 1 }, 2);
	Tensor<float> t3 = fill<float>({ 5, 1 }, 0);
	std::cout << t2 << std::endl;

	//matmul(t1, t2, t3, cublasHandle);
	//std::cout << t3 << std::endl;

	sigmoid(t3, t3, cudnnHandle);
	std::cout << t3 << std::endl;

	cudaFree(t1.begin());
}
