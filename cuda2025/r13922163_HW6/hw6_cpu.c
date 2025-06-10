#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 81920000

const int NUM_BINS = 1024;
const float MAX_X = 32.0f;
const float BIN_WIDTH = MAX_X / NUM_BINS;

void histo_kernel(float *data, int size, unsigned int *histo)
{
    for (int i = 0; i < size; ++i) {
        int bin = (data[i] < MAX_X) ? (int)(data[i] / BIN_WIDTH) : NUM_BINS - 1;
        histo[bin]++;
    }
}

float generate_exponential()
{
    float u = (rand() + 1.0f) / (RAND_MAX + 2.0f); // Avoid log(0)
    return -log(u);
}

float *big_random_block(int size)
{
    float *data = (float *)malloc(sizeof(float) * size);
    for (int i = 0; i < size; ++i) {
        data[i] = generate_exponential();
    }
    return data;
}

int main()
{
    float *data = big_random_block(SIZE);

    unsigned int histo[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++)
        histo[i] = 0;

    clock_t start = clock();
    histo_kernel(data, SIZE, histo);
    clock_t end = clock();

    double elapsed_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    printf("Time to generate: %.2f ms\n", elapsed_ms);

    long histoCount = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        histoCount += histo[i];
    }
    printf("Histogram Sum:  %ld\n", histoCount);

    FILE *f = fopen("histogram_cpu.txt", "w");
    for (int i = 0; i < NUM_BINS; ++i)
        fprintf(f, "%f\t%u\n", i * BIN_WIDTH + 0.5f * BIN_WIDTH, histo[i]);
    fclose(f);

    // Verification
    for (int i = 0; i < SIZE; i++) {
        int bin = (data[i] < MAX_X) ? (int)(data[i] / BIN_WIDTH) : NUM_BINS - 1;
        histo[bin]--;
    }
    for (int i = 0; i < NUM_BINS; i++) {
        if (histo[i] != 0)
            printf("Failure at %d! Off by %d\n", i, histo[i]);
    }

    free(data);
    return 0;
}
