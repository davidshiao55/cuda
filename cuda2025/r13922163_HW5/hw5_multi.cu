/**********************************************************************
 * hw5_multi.cu  – 2-D Jacobi on multiple GPUs (one pthread per GPU)
 * Usage : ./jacobi <maxIter> <Tblock> [nGPU]
 *   maxIter : # Jacobi iterations
 *   Tblock  : square CUDA block size (≤32 so 32×32 ≤1024 thr/block)
 *   nGPU    : optional, # GPUs to use (default = all visible)
 * Output : output_multi.csv  (1024×1024 interior temps)
 *********************************************************************/

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------------- stencil kernels ---------------- */

__global__ void blend_kernel(float *out, const float *in, int DIM, int ROWS)
{
    int N = DIM + 2;
    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x; // 1…DIM
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y; // 1…ROWS
    if (i > DIM || j > ROWS)
        return;

    int idx = i + j * N;
    out[idx] = 0.25f * (in[idx - 1] + in[idx + 1] +
                        in[idx - N] + in[idx + N]);
}

__global__ void copy_const_kernel(float *field, const float *fixed,
                                  int DIM, int ROWS)
{
    int N = DIM + 2;
    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i > DIM || j > ROWS)
        return;

    int idx = i + j * N;
    if (i == 1 || i == DIM || j == 1 || j == ROWS)
        field[idx] = fixed[idx];
}

/* ---------------- per-GPU worker ---------------- */

typedef struct {
    int id;
    int rows;
    int dim;
    float *d_in, *d_out, *d_fix;
    cudaStream_t sComp, sComm;
    cudaEvent_t evt0, evt1;
    double time_ms; // accumulated time
    pthread_barrier_t *bar;
} Worker;

/* ---------------- globals ---------------- */

static Worker *gWork = NULL;
static size_t pitchB = 0; // bytes per row
static int N_GPU = 0;
static int MAXITER = 0;
static int T_BLOCK = 0;

/* ---------------- thread routine ---------------- */

static void *routine(void *arg)
{
    Worker *w = (Worker *)arg;
    cudaSetDevice(w->id);

    cudaStreamCreate(&w->sComp);
    cudaStreamCreate(&w->sComm);
    cudaEventCreate(&w->evt0);
    cudaEventCreate(&w->evt1);

    dim3 threads(T_BLOCK, T_BLOCK);
    dim3 grid((w->dim + T_BLOCK - 1) / T_BLOCK,
              (w->rows + T_BLOCK - 1) / T_BLOCK);

    for (int it = 0; it < MAXITER; ++it) {

        /* --- start timer on THIS GPU --- */
        cudaEventRecord(w->evt0, w->sComp);

        /* --- compute interior + overwrite edges --- */
        blend_kernel<<<grid, threads, 0, w->sComp>>>(w->d_out, w->d_in,
                                                     w->dim, w->rows);
        copy_const_kernel<<<grid, threads, 0, w->sComp>>>(w->d_out, w->d_fix,
                                                          w->dim, w->rows);

        /* signal sComm when compute done */
        cudaEventRecord(w->evt1, w->sComp);
        cudaStreamWaitEvent(w->sComm, w->evt1, 0);

        /* --- halo exchange --- */
        size_t rowBytes = pitchB;
        int up = w->id - 1, dn = w->id + 1;
        if (up >= 0)
            cudaMemcpyPeerAsync(
                gWork[up].d_in + (gWork[up].rows + 1) * (w->dim + 2), up,
                w->d_out + 1 * (w->dim + 2), w->id,
                rowBytes, w->sComm);

        if (dn < N_GPU)
            cudaMemcpyPeerAsync(
                gWork[dn].d_in, dn,
                w->d_out + w->rows * (w->dim + 2), w->id,
                rowBytes, w->sComm);

        cudaStreamSynchronize(w->sComm); // halos arrived

        /* --- stop timer on THIS GPU --- */
        cudaEventRecord(w->evt1, w->sComm);
        cudaEventSynchronize(w->evt1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, w->evt0, w->evt1);
        w->time_ms += ms;

        /* swap & global barrier */
        float *tmp = w->d_in;
        w->d_in = w->d_out;
        w->d_out = tmp;
        pthread_barrier_wait(w->bar);
    }
    return NULL;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s <maxIter> <Tblock> [nGPU]\n", argv[0]);
        return 0;
    }
    MAXITER = atoi(argv[1]);
    T_BLOCK = atoi(argv[2]);
    if (T_BLOCK < 1)
        T_BLOCK = 1;
    if (T_BLOCK > 32)
        T_BLOCK = 32; // 32×32=1024

    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    N_GPU = (argc >= 4) ? atoi(argv[3]) : devCount;
    if (N_GPU < 1 || N_GPU > devCount)
        N_GPU = devCount;

    const int DIM = 1024;
    const int Nside = DIM + 2;
    pitchB = Nside * sizeof(float);

    printf("#GPU   = %d\niters  = %d\nblock  = %dx%d\n",
           N_GPU, MAXITER, T_BLOCK, T_BLOCK);

    /* --- init full plate on host ------------------------------- */
    size_t Ncells = (size_t)Nside * Nside;
    float *h_full = (float *)malloc(Ncells * sizeof(float));
    for (size_t k = 0; k < Ncells; ++k)
        h_full[k] = 273.f;
    for (int i = 1; i <= DIM; ++i)
        h_full[i + Nside] = 400.f; // hot top

    /* --- split rows & allocate device slabs -------------------- */
    gWork = (Worker *)calloc(N_GPU, sizeof(Worker));
    int base = DIM / N_GPU, extra = DIM % N_GPU;

    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, N_GPU);

    int gRow = 0;
    for (int id = 0; id < N_GPU; ++id) {
        Worker *w = &gWork[id];
        w->id = id;
        w->rows = base + (id < extra ? 1 : 0);
        w->dim = DIM;
        w->bar = &bar;
        w->time_ms = 0.0;

        cudaSetDevice(id);
        size_t slabCells = (size_t)(w->rows + 2) * Nside;
        cudaMalloc(&w->d_in, slabCells * sizeof(float));
        cudaMalloc(&w->d_out, slabCells * sizeof(float));
        cudaMalloc(&w->d_fix, slabCells * sizeof(float));

        cudaMemcpy2D(w->d_in, pitchB,
                     h_full + gRow * Nside, pitchB,
                     pitchB, w->rows + 2,
                     cudaMemcpyHostToDevice);
        cudaMemcpy(w->d_out, w->d_in, slabCells * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(w->d_fix, w->d_in, slabCells * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        gRow += w->rows;
    }

    /* --- enable peer access where possible --------------------- */
    for (int a = 0; a < N_GPU; ++a) {
        cudaSetDevice(a);
        for (int b = 0; b < N_GPU; ++b)
            if (a != b)
                cudaDeviceEnablePeerAccess(b, 0);
    }

    /* --- launch persistent threads ----------------------------- */
    pthread_t *tid = (pthread_t *)malloc(N_GPU * sizeof(pthread_t));
    for (int id = 0; id < N_GPU; ++id)
        pthread_create(&tid[id], NULL, routine, &gWork[id]);
    for (int id = 0; id < N_GPU; ++id)
        pthread_join(tid[id], NULL);

    /* --- ensure all GPU work done ------------------------------ */
    for (int id = 0; id < N_GPU; ++id) {
        cudaSetDevice(id);
        cudaDeviceSynchronize();
    }

    /* --- gather interior back to host -------------------------- */
    gRow = 0;
    for (int id = 0; id < N_GPU; ++id) {
        Worker *w = &gWork[id];
        cudaSetDevice(id);
        cudaMemcpy2D(h_full + (gRow + 1) * Nside, pitchB,
                     w->d_in + Nside, pitchB,
                     pitchB, w->rows,
                     cudaMemcpyDeviceToHost);
        gRow += w->rows;
    }

    /* --- CSV output -------------------------------------------- */
    FILE *fp = fopen("output_multi.csv", "w");
    for (int j = 1; j <= DIM; ++j) {
        for (int i = 1; i <= DIM; ++i) {
            fprintf(fp, "%f%s", h_full[i + j * Nside],
                    (i == DIM ? "" : ","));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Result save to output_multi.csv\n");

    /* --- timing report ----------------------------------------- */
    double max_ms = 0.0;
    for (int id = 0; id < N_GPU; ++id) {
        double avg = gWork[id].time_ms / MAXITER;
        if (avg > max_ms)
            max_ms = avg;
    }
    printf("avg time / iteration = %.6f  ms\n", max_ms);

    /* --- cleanup ----------------------------------------------- */
    for (int id = 0; id < N_GPU; ++id) {
        cudaSetDevice(id);
        cudaFree(gWork[id].d_in);
        cudaFree(gWork[id].d_out);
        cudaFree(gWork[id].d_fix);
        cudaStreamDestroy(gWork[id].sComp);
        cudaStreamDestroy(gWork[id].sComm);
        cudaEventDestroy(gWork[id].evt0);
        cudaEventDestroy(gWork[id].evt1);
    }
    free(tid);
    free(h_full);
    free(gWork);
    return 0;
}
