
#include "cudaLib.cuh"
#include <cstdlib>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		y[i] = scale * x[i] + y[i];
	}
}

int runGpuSaxpy(int vectorSize) {
	cudaError_t err;

	// random alpha value
	float alpha = (float)(rand() % 100);

	// Allocate host memory (X_h, Y_h, YResult_h)
	float *X_h = new float[vectorSize]; 
	float *Y_h = new float[vectorSize]; 
	float *YResult_h = new float[vectorSize]; 

	// Populate arrays X_h and Y_h with random values
	vectorInit(X_h, vectorSize);
	vectorInit(Y_h, vectorSize);

	int size = vectorSize * sizeof(float);
	// Allocate device memory two arrays of size vectorSize (X_d, Y_d)
	float *X_d, *Y_d;

	err=cudaMalloc((void **) &X_d, size);
	gpuAssert(err, __FILE__, __LINE__, false);
	err=cudaMalloc((void **) &Y_d, size);
	gpuAssert(err, __FILE__, __LINE__, false);

	// Move values to device
	err=cudaMemcpy(X_d, X_h, size, cudaMemcpyHostToDevice);
	gpuAssert(err, __FILE__, __LINE__, false);
	err=cudaMemcpy(Y_d, Y_h, size, cudaMemcpyHostToDevice);
	gpuAssert(err, __FILE__, __LINE__, false);

	// invoke kernel
	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(X_d, Y_d, alpha, vectorSize);

	// Move results from device to host
	err=cudaMemcpy(YResult_h, Y_d, size, cudaMemcpyDeviceToHost);
	gpuAssert(err, __FILE__, __LINE__, false);

	// Check Result
	int errorCount = verifyVector(X_h, Y_h, YResult_h, alpha, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// Free device memory (X_d, Y_d)
	cudaFree(X_d);
	cudaFree(Y_d);

	// Free host memory (X_h, Y_h, YResult_h)
	delete[] X_h;
	delete[] Y_h;
	delete[] YResult_h;

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	uint64_t hits = 0;
	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);
	if (i < pSumSize)
	{
		for (int j = 0; j<sampleSize; j++)
		{
			float x = curand_uniform(&rng);
        	float y = curand_uniform(&rng);

			if ( int(x * x + y * y) == 0 ) {
				++ hits;
			}
		}
		pSums[i] = hits;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	cudaError_t err;
	
	double approxPi = 0;
	uint64_t hits = 0;

	uint64_t size = generateThreadCount * sizeof(uint64_t);

	// Allocate host memory (pSums_h)
	uint64_t *pSums_h = new uint64_t[generateThreadCount];

	// Allocate device memory (pSums_d)
	uint64_t *pSums_d;
	err=cudaMalloc((void **) &pSums_d, size);
	gpuAssert(err, __FILE__, __LINE__, false);

	// invoke kernel
	generatePoints<<<ceil(generateThreadCount/256.0), 256>>>(pSums_d, generateThreadCount, sampleSize);

	// Move results from device to host
	err=cudaMemcpy(pSums_h, pSums_d, size, cudaMemcpyDeviceToHost);
	gpuAssert(err, __FILE__, __LINE__, false);

	// Sum total hits over pSums_h
	for (uint64_t i=0;i<generateThreadCount;i++)
	{
		hits += pSums_h[i];
	}
	approxPi = 4.0*(hits/static_cast<double>(generateThreadCount*sampleSize));

	// Free device memory (X_d, Y_d)
	cudaFree(pSums_d);

	// Free host memory (X_h, Y_h, YResult_h)
	delete[] pSums_h;

	return approxPi;
}