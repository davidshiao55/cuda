#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define imin(a, b) (a < b ? a : b)

const int N = 40960000;
int threadsPerBlock;
int blocksPerGrid;

__global__ void dot(int size, float *a, float *b, float *c)
{
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float tmp = 0;
    while (tid < size) {
        tmp += a[tid] * b[tid];
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

struct DataStruct {
    int deviceID;
    int size;
    float *a;
    float *b;
    float returnValue;
    float elapsedTime;
};

void *routine(void *pvoidData)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    DataStruct *data = (DataStruct *)pvoidData;
    cudaSetDevice(data->deviceID);

    int size = data->size;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = data->a;
    b = data->b;
    partial_c = (float *)malloc(sizeof(float) * blocksPerGrid);

    // allocate the memory on the GPU
    cudaMalloc(&dev_a, sizeof(float) * size);
    cudaMalloc(&dev_b, sizeof(float) * size);
    cudaMalloc(&dev_partial_c, sizeof(float) * blocksPerGrid);

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    // captrue the start time
    cudaEventRecord(start, 0);
    dot<<<blocksPerGrid, threadsPerBlock, sizeof(float) * threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
    // get stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // copy the arrays 'partial_c' back to the CPU
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    // free memory on GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    // free memory on CPU
    free(partial_c);

    data->returnValue = c;
    data->elapsedTime = gpu_time;
    pthread_exit(NULL);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: %s <n_gpus> <threadsPerBlock> <blocksPerGrid>\n", argv[0]);
        return 1;
    }
    const int n_gpus = atoi(argv[1]);
    threadsPerBlock = atoi(argv[2]);
    blocksPerGrid = atoi(argv[3]);

    printf("n_gpus = %d\n", n_gpus);
    printf("threadsPerBlock = %d\n", threadsPerBlock);
    printf("blocksPerGrid = %d\n", blocksPerGrid);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < n_gpus) {
        printf("We need at least %d compute 1.0 or greater devices, but only found %d\n", n_gpus, deviceCount);
        return 0;
    }

    cudaDeviceProp prop;
    for (int i = 0; i < n_gpus; i++) {
        cudaGetDeviceProperties(&prop, i);
        if (prop.canMapHostMemory != 1) {
            printf("Device %d cannot map memory.\n", i);
            return 0;
        }
    }

    float *a, *b, c;
    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);

    // fill a & b with random number between 0.0 and 1.0
    for (int i = 0; i < N; i++) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    DataStruct data[n_gpus];

    for (int i = 0; i < n_gpus; i++) {
        data[i].deviceID = i;
        data[i].size = N / n_gpus;
        data[i].a = a + N / n_gpus * i;
        data[i].b = b + N / n_gpus * i;
    }

    pthread_t ts[n_gpus];
    for (int i = 0; i < n_gpus; ++i)
        pthread_create(&ts[i], NULL, routine, (void *)&data[i]);
    for (int i = 0; i < n_gpus; ++i)
        pthread_join(ts[i], NULL);

    c = 0.0f;
    for (int i = 0; i < n_gpus; ++i)
        c += data[i].returnValue;

    float gpu_time = 0;
    for (int i = 0; i < n_gpus; i++)
        gpu_time = max(gpu_time, data[i].elapsedTime);
    printf("GPU elapsed time: %3.1f ms\n", gpu_time);

    double c_cpu = 0;
    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++) {
        c_cpu += (double)a[i] * b[i];
    }
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU elapsed time: %3.5f ms\n", cpu_time);
    printf("Speed up = %3.1f\n", cpu_time / gpu_time);

    if (fabs(c_cpu - (double)c) < 1e-5 * fabs(c_cpu)) /* NEW – sense reversed       */
        printf("Results match!\n");
    else
        printf("Mismatch! CPU %.8e  GPU %.8e\n", c_cpu, c);

    cudaFreeHost(a);
    cudaFreeHost(b);

    return 0;
}