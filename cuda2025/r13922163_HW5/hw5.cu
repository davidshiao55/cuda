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

    // four neighbours (all legal because of ghost layer)
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

    printf("maxiter = %d\n", maxiter);
    printf("threadsPerBlock = (%d, %d)\n", T, T);

    size_t N = Nside * Nside; // include ghost cell to avoid branch divergence
    float *h_src = (float *)malloc(N * sizeof(float));
    for (int j = 0; j < Nside; ++j)
        for (int i = 0; i < Nside; ++i) {
            int idx = i + j * Nside;
            h_src[idx] = 273.f;
        }
    // set physical top edge (row 1) to 400 K
    for (int i = 1; i <= DIM; ++i)
        h_src[i + Nside] = 400.f;

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

        blend_kernel<<<blocks, threads>>>(d_out, d_in, DIM); // write "out" from "in"
        copy_const_kernel<<<blocks, threads>>>(d_out, d_src, DIM);

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
    printf("gpu time per update: %3.5f ms\n", total_time / maxiter);

    cudaMemcpy(h_src, d_in, sizeof(float) * N, cudaMemcpyDeviceToHost);

    FILE *fp = fopen("output.csv", "w");
    for (int j = 1; j <= DIM; ++j) {
        for (int i = 1; i <= DIM; ++i) {
            int idx = i + j * Nside;
            fprintf(fp, "%f", h_src[idx]);
            if (i != DIM)
                fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Result save to output.csv\n");

    free(h_src);
    cudaFree(d_src);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}