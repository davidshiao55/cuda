#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s <maxiter>\n", argv[0]);
        return 1;
    }

    int const maxiter = atoi(argv[1]);
    int const DIM = 1024;
    int const Nside = DIM + 2;

    printf("maxiter = %d\n", maxiter);

    size_t N = Nside * Nside; // include ghost cell to avoid branch divergence
    float *src = (float *)malloc(N * sizeof(float));
    float *in = (float *)malloc(N * sizeof(float));
    float *out = (float *)malloc(N * sizeof(float));
    for (int j = 0; j < Nside; ++j)
        for (int i = 0; i < Nside; ++i) {
            int idx = i + j * Nside;
            src[idx] = 273.f;
            in[idx] = 273.f;
            out[idx] = 273.f;
        }

    // set physical top edge (row 1) to 400 K
    for (int i = 1; i <= DIM; ++i) {
        src[i + Nside] = 400.f;
        in[i + Nside] = 400.f;
        out[i + Nside] = 400.f;
    }

    float total_time = 0.f;
    for (int k = 0; k < maxiter; k++) {
        clock_t cpu_start = clock();
        // blend
        for (int j = 1; j <= DIM; j++) {
            for (int i = 1; i <= DIM; i++) {
                int idx = i + j * Nside;

                // four neighbours (all legal because of ghost layer)
                float left = in[idx - 1];
                float right = in[idx + 1];
                float back = in[idx - Nside];
                float front = in[idx + Nside];

                out[idx] = (left + right + back + front) * (1.0f / 4.0f);
            }
        }
        float *tmp = out;
        // copy const
        for (int j = 1; j <= DIM; j++) {
            for (int i = 1; i <= DIM; i++) {
                int idx = i + j * Nside;
                if (i == 1 || i == DIM || j == 1 || j == DIM) // edge
                    out[idx] = src[idx];
            }
        }
        out = in;
        in = tmp;
        clock_t cpu_end = clock();
        float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
        total_time += cpu_time;
    }
    printf("avg time / iteration = %.6f  ms\n", total_time / maxiter);

    FILE *fp = fopen("output_cpu.csv", "w");
    for (int j = 1; j <= DIM; ++j) {
        for (int i = 1; i <= DIM; ++i) {
            int idx = i + j * Nside;
            fprintf(fp, "%f", in[idx]);
            if (i != DIM)
                fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    free(src);
    free(in);
    free(out);
    printf("Result save to output_cpu.csv\n");
    return 0;
}