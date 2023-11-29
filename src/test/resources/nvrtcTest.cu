#define N 1000

extern "C" __global__ void matSum(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

extern "C" __global__ void helloCuda(int *a)
{
    int tid = blockIdx.x;
    if (tid == 0)
        printf("helloCuda on thread %d value: %d\n", tid, a[0]);
}
