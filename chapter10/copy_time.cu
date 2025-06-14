#include "../common/book.h"

#define SIZE (10 * 1024 * 1024)

float cuda_malloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (int *)malloc(sizeof(int) * size);
    HANDLE_NULL(a);
    HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(int) * size));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    for (int i = 0; i < 100; i++) {
        if (up)
            HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(cudaMemcpy(a, dev_a, sizeof(int) * size, cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

float cuda_host_alloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaHostAlloc(&a, sizeof(int) * size, cudaHostAllocDefault));
    HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(int) * size));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    for (int i = 0; i < 100; i++) {
        if (up)
            HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(cudaMemcpy(a, dev_a, sizeof(int) * size, cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

int main()
{
    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    elapsedTime = cuda_malloc_test(SIZE, true);
    printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_malloc_test(SIZE, false);
    printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, true);
    printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, false);
    printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f\n", MB / (elapsedTime / 1000));
    return 0;
}