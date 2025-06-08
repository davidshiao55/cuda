#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 6400

__global__ void add(float *a, float *b, float *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N * N) {
        c[tid] = 1 / a[tid] + 1 / b[tid];
        tid += blockDim.x * gridDim.x;
    }
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

    float *dev_a, *dev_b, *dev_c;
    float *a, *b, *c;
    // allocate memory on cpu
    a = (float *)malloc(sizeof(float) * N * N);
    b = (float *)malloc(sizeof(float) * N * N);
    c = (float *)malloc(sizeof(float) * N * N);
    // fill a & b with random number between 0.0 and 1.0
    for (int i = 0; i < N * N; i++) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    // allocate memory on gpu
    cudaMalloc((void **)&dev_a, sizeof(float) * N * N);
    cudaMalloc((void **)&dev_b, sizeof(float) * N * N);
    cudaMalloc((void **)&dev_c, sizeof(float) * N * N);

    // copy a & b to gpu
    cudaMemcpy(dev_a, a, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // captrue the start time
    cudaEventRecord(start, 0);
    add<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);
    // get stop time, and display timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU elapsed time: %3.1f ms\n", gpu_time);

    // copy the array c back to cpu
    cudaMemcpy(c, dev_c, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    // CPU reference
    float *c_cpu = (float *)malloc(sizeof(float) * N * N);
    clock_t cpu_start = clock();
    for (int i = 0; i < N * N; i++) {
        c_cpu[i] = 1.0f / a[i] + 1.0f / b[i];
    }
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU elapsed time: %3.1f ms\n", cpu_time);
    printf("Speed up = %3.1f\n", cpu_time / gpu_time);

    // verify that the gpu did the work we requested
    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(c_cpu[i] - c[i]) > 1e-5) {
            printf("Error: 1/%f + 1/%f != %f\n", a[i], b[i], c[i]);
            success = false;
            break;
        }
    }
    if (success)
        printf("Correct Result!!!\n");

    // free memory allocated on the gpu
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    // free memory allocated on the cpu
    free(a);
    free(b);
    free(c);
    free(c_cpu);

    return 0;
}