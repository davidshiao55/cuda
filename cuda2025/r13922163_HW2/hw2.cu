#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int N = 6400;

__global__ void trace(float *a, float *c)
{
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int cacheIndex = threadIdx.x;
    float tmp = 0;

    while (tid < N) {
        tmp += a[tid * N + tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache value
    cache[cacheIndex] = tmp;

    // synchronize threads in this block
    __syncthreads();

    // reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <threadsPerBlock> <blocksPerGrid>\n", argv[0]);
        return 1;
    }

    const int threadsPerBlock = atoi(argv[1]);
    const int blocksPerGrid = atoi(argv[2]);
    printf("threadsPerBlock = %d\n", threadsPerBlock);
    printf("blocksPerGrid = %d\n", blocksPerGrid);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *a, c, *partial_c;
    float *dev_a, *dev_partial_c;

    // allocate memory on the CPU
    a = (float *)malloc(sizeof(float) * N * N);
    partial_c = (float *)malloc(sizeof(float) * blocksPerGrid);

    // fill a & b with random number between 0.0 and 1.0
    for (int i = 0; i < N * N; i++) {
        a[i] = (float)rand() / (float)RAND_MAX;
    }

    // allocate memory on the GPU
    cudaMalloc((void **)&dev_a, sizeof(float) * N * N);
    cudaMalloc((void **)&dev_partial_c, sizeof(float) * blocksPerGrid);

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // captrue the start time
    cudaEventRecord(start, 0);
    trace<<<blocksPerGrid, threadsPerBlock, sizeof(float) * threadsPerBlock>>>(dev_a, dev_partial_c);
    // get stop time, and display timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU elapsed time: %3.5f ms\n", gpu_time);

    // copy the arrays 'partial_c' back to the CPU
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        // printf("%f\n", partial_c[i]);
        c += partial_c[i];
    }

    float c_cpu = 0;
    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++) {
        c_cpu += a[N * i + i];
    }
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU elapsed time: %3.5f ms\n", cpu_time);
    printf("Speed up = %3.1f\n", cpu_time / gpu_time);

    if (fabs(c_cpu - c) < 1e-5 * fabs(c_cpu))
        printf("Results match!\n");
    else
        printf("Mismatch! CPU %.8e  GPU %.8e\n", c_cpu, c);

    // free memory on GPU
    cudaFree(dev_a);
    cudaFree(dev_partial_c);

    // free memory on CPU
    free(a);
    free(partial_c);
}