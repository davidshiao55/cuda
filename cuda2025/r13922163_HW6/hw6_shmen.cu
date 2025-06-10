#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 81920000

const int NUM_BINS = 1024;
const float MAX_X = 32.0f;
const float BIN_WIDTH = MAX_X / NUM_BINS;

__global__ void histo_kernel(float *data, int size, unsigned int *histo)
{
    __shared__ unsigned int temp[NUM_BINS]; // One shared histogram per block

    // Initialize shared histogram
    int tid = threadIdx.x;
    for (int i = tid; i < NUM_BINS; i += blockDim.x)
        temp[i] = 0;
    __syncthreads();

    // Global thread index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        int bin = (data[i] < MAX_X) ? (int)(data[i] / BIN_WIDTH) : NUM_BINS - 1;
        atomicAdd(&temp[bin], 1); // Shared memory accumulation
        i += stride;
    }

    __syncthreads();

    // Write shared histogram to global memory
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&(histo[i]), temp[i]);
    }
}

float generate_exponential()
{
    float u = (rand() + 1.0) / (RAND_MAX + 2.0); // Avoid log(0)
    return -log(u);
}

float *big_random_block(int size)
{
    float *data = (float *)malloc(sizeof(float) * size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = generate_exponential();
    }

    return data;
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

    float *data = big_random_block(SIZE);

    // capture the start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on the GPU for the file's data
    float *dev_data;
    unsigned int *dev_histo;
    cudaMalloc((void **)&dev_data, sizeof(float) * SIZE);
    cudaMemcpy(dev_data, data, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_histo, NUM_BINS * sizeof(int));
    cudaMemset(dev_histo, 0, NUM_BINS * sizeof(int));

    cudaEventRecord(start, 0);
    histo_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_data, SIZE, dev_histo);
    // get stop time, and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    unsigned int histo[NUM_BINS];
    cudaMemcpy(histo, dev_histo, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    long histoCount = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        histoCount += histo[i];
    }
    printf("Histogram Sum:  %ld\n", histoCount);

    FILE *f = fopen("histogram.txt", "w");
    for (int i = 0; i < NUM_BINS; ++i)
        fprintf(f, "%f\t%u\n", i * BIN_WIDTH + 0.5 * BIN_WIDTH, histo[i]);
    fclose(f);

    // verify that we have the same counts via CPU
    for (int i = 0; i < SIZE; i++) {
        int bin = (data[i] < MAX_X) ? (int)(data[i] / BIN_WIDTH) : NUM_BINS - 1; // last bin for overflow
        histo[bin]--;
    }
    for (int i = 0; i < NUM_BINS; i++) {
        if (histo[i] != 0)
            printf("Failure at %d!  Off by %d\n", i, histo[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_histo);
    cudaFree(dev_data);
    free(data);
    return 0;
}
