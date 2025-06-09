#include "../common/book.h"

#define imin(a, b) (a < b ? a : b)
#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int size, float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
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

float malloc_test(int size)
{
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // allocate memory on the CPU side
    a = (float *)malloc(sizeof(float) * size);
    b = (float *)malloc(sizeof(float) * size);
    partial_c = (float *)malloc(sizeof(float) * blocksPerGrid);

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(float) * size));
    HANDLE_ERROR(cudaMalloc(&dev_b, sizeof(float) * size));
    HANDLE_ERROR(cudaMalloc(&dev_partial_c, sizeof(float) * blocksPerGrid));

    // fill in the host memory with data
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaEventRecord(start, 0));
    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    // copy the arrays 'partial_c' back to the CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

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
    free(a);
    free(b);
    free(partial_c);

    // free Event
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    printf("Value calculated: %f\n", c);

    return elapsedTime;
}

float cuda_host_alloc_test(int size)
{
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // allocate memory on the CPU side
    HANDLE_ERROR(cudaHostAlloc(&a, sizeof(float) * size, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc(&b, sizeof(float) * size, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc(&partial_c, sizeof(float) * blocksPerGrid, cudaHostAllocMapped));

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

    // fill in the host memory with data
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaEventRecord(start, 0));
    // copy the arrays 'a' and 'b' to the GPU

    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
    HANDLE_ERROR(cudaThreadSynchronize());

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(partial_c));

    // free Event
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    printf("Value calculated: %f\n", c);

    return elapsedTime;
}

int main()
{
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
    if (prop.canMapHostMemory != 1) {
        printf("Device cannot map memory\n");
        return 0;
    }
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

    float elapsedTime = malloc_test(N);
    printf("Time using cudaMalloc %3.5f ms\n", elapsedTime);

    elapsedTime = cuda_host_alloc_test(N);
    printf("Time using cudaHostAlloc: %3.5f ms\n", elapsedTime);
}