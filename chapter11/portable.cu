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

struct DataStruct {
    int deviceID;
    int offset;
    int size;
    float *a;
    float *b;
    float returnValue;
};

void *routine(void *pvoidData)
{
    DataStruct *data = (DataStruct *)pvoidData;
    if (data->deviceID != 0) {
        HANDLE_ERROR(cudaSetDevice(data->deviceID));
        HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
    }
    HANDLE_ERROR(cudaSetDevice(data->deviceID));

    int size = data->size;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = data->a;
    b = data->b;
    partial_c = (float *)malloc(sizeof(float) * blocksPerGrid);

    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
    HANDLE_ERROR(cudaMalloc(&dev_partial_c, sizeof(float) * blocksPerGrid));

    // offset 'a' and 'b' to where this GPU gets it data
    dev_a += data->offset;
    dev_b += data->offset;

    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    // copy the arrays 'partial_c' back to the CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    // free memory on GPU
    HANDLE_ERROR(cudaFree(dev_partial_c));

    // free memory on CPU
    free(partial_c);

    data->returnValue = c;
    return 0;
}

int main()
{
    int deviceCount;
    HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
        return 0;
    }

    cudaDeviceProp prop;
    for (int i = 0; i < 2; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        if (prop.canMapHostMemory != 1) {
            printf("Device %d cannot map memory.\n", i);
            return 0;
        }
    }

    float *a, *b;
    HANDLE_ERROR(cudaSetDevice(0));
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
    HANDLE_ERROR(cudaHostAlloc(&a, sizeof(float) * N, cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc(&b, sizeof(float) * N, cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped));

    // fill in the host memory with data
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    DataStruct data[2];

    data[0].deviceID = 0;
    data[0].offset = 0;
    data[0].size = N / 2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].offset = N / 2;
    data[1].size = N / 2;
    data[1].a = a;
    data[1].b = b;

    CUTThread thread = start_thread(routine, &(data[0]));
    routine(&(data[1]));
    end_thread(thread);

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));

    printf("Value calculated: %f\n", data[0].returnValue + data[1].returnValue);
    return 0;
}