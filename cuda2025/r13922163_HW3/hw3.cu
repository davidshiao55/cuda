#include <stdio.h>
#include <stdlib.h>

#define T 8

__global__ void poisson(float *out, const float *in, int L, int idxCenter)
{
    int N = L + 2; // physical row length (ghosts included)
    int pitchY = N;
    int pitchZ = N * N;

    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = 1 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i > L || j > L || k > L)
        return;

    int idx = i + j * pitchY + k * pitchZ;

    // six neighbours (all legal because of ghost layer)
    float left = in[idx - 1];
    float right = in[idx + 1];
    float back = in[idx - pitchY];
    float front = in[idx + pitchY];
    float up = in[idx - pitchZ];
    float down = in[idx + pitchZ];

    float newVal = (left + right + back + front + up + down) * (1.0f / 6.0f);
    if (idx == idxCenter)
        newVal += 1.0f / 6.0f; // ρ term
    out[idx] = newVal;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <L> <maxiter>\n", argv[0]);
        return 1;
    }
    int const L = atoi(argv[1]);
    int const maxiter = atoi(argv[2]);
    int const Nside = L + 2;
    int const mid = L / 2 + 1;

    size_t N = Nside * Nside * Nside; // include ghost cell to avoid branch divergence
    float *source = (float *)malloc(N * sizeof(float));
    memset(source, 0, N * sizeof(float));
    int idxCenter = mid + mid * Nside + mid * Nside * Nside;

    float *out, *in;
    cudaMalloc(&out, N * sizeof(float));
    cudaMalloc(&in, N * sizeof(float));
    cudaMemcpy(in, source, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out, source, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks((L + T - 1) / T, (L + T - 1) / T, (L + T - 1) / T);
    dim3 threads(T, T, T);

    for (int iter = 0; iter < maxiter; ++iter) {
        poisson<<<blocks, threads>>>(out, in, L, idxCenter); // write "out" from "in"
        std::swap(in, out);                                  // next step will read the new “in”
    }

    cudaMemcpy(source, in, sizeof(float) * N, cudaMemcpyDeviceToHost);
    double sum[L + 1] = {0};
    int cnt[L + 1] = {0};

    for (int z = 1; z < L + 1; z++) {
        for (int y = 1; y < L + 1; y++) {
            for (int x = 1; x < L + 1; x++) {
                int idx = x + y * Nside + z * Nside * Nside;
                int dx = x - mid, dy = y - mid, dz = z - mid;
                int r = int(llround(std::sqrt(dx * dx + dy * dy + dz * dz)));
                if (r == 0 || r > L)
                    continue; // skip centre & oversized bins
                sum[r] += source[idx];
                cnt[r] += 1;
            }
        }
    }

    printf("# r   phi_avg(L=%d)\n", L);
    for (int r = 1; r <= L; ++r)
        if (cnt[r]) // avoid divide-by-zero
            printf("%2d %.7e\n", r, sum[r] / cnt[r]);

    free(source);
    cudaFree(in);
    cudaFree(out);
    return 0;
}