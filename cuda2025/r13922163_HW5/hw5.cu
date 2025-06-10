#include <stdio.h>
#include <stdlib.h>

__global__ void blend_kernel(float *out, const float *in, int DIM)
{
    int N = DIM + 2; // physical row length (ghosts included)
    int pitchY = N;

    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i > DIM || j > DIM)
        return;

    int idx = i + j * pitchY;

    // six neighbours (all legal because of ghost layer)
    float left = in[idx - 1];
    float right = in[idx + 1];
    float back = in[idx - pitchY];
    float front = in[idx + pitchY];

    out[idx] = (left + right + back + front) * (1.0f / 4.0f);
}

__global__ void copy_const_kernel(float *iptr, float *source, int DIM)
{
    int N = DIM + 2; // physical row length (ghosts included)
    int pitchY = N;

    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i > DIM || j > DIM)
        return;

    int idx = i + j * pitchY;
    if (i == 1 || i == DIM || j == 1 || j == DIM) // edge
        iptr[idx] = source[idx];
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <maxiter> <T>\n", argv[0]);
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int const DIM = 1024;
    int const maxiter = atoi(argv[1]);
    int const T = atoi(argv[2]);
    int const Nside = DIM + 2;

    size_t N = Nside * Nside; // include ghost cell to avoid branch divergence
    float *h_src = (float *)malloc(N * sizeof(float));
    for (int j = 0; j < Nside; ++j)
        for (int i = 0; i < Nside; ++i) {
            int idx = i + j * Nside;
            h_src[idx] = 273.f;
        }
    // set physical top edge (row DIM) to 400 K
    for (int i = 1; i <= DIM; ++i)
        h_src[i + DIM * Nside] = 400.f;

    float *d_out, *d_in, *d_src;
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_src, N * sizeof(float));

    cudaMemcpy(d_in, h_src, N * sizeof(float), cudaMemcpyHostToDevice);  // interior+edges
    cudaMemcpy(d_out, h_src, N * sizeof(float), cudaMemcpyHostToDevice); // safe first swap
    cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice); // immutable boundary

    dim3 blocks((DIM + T - 1) / T, (DIM + T - 1) / T);
    dim3 threads(T, T);

    float total_time = 0;
    for (int iter = 0; iter < maxiter; ++iter) {
        // captrue the start time
        cudaEventRecord(start, 0);

        copy_const_kernel<<<blocks, threads>>>(d_in, d_src, DIM);
        blend_kernel<<<blocks, threads>>>(d_out, d_in, DIM); // write "out" from "in"

        // get stop time, and display timing results
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start, stop);
        total_time += gpu_time;

        float *tmp = d_out;
        d_out = d_in;
        d_in = tmp;
    }
    printf("time per update: %3.5f ms\n", total_time / maxiter);

    cudaMemcpy(h_src, d_in, sizeof(float) * N, cudaMemcpyDeviceToHost);

    free(h_src);
    cudaFree(d_src);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}