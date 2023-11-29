extern "C" __global__ void sum(int N, int *a, int *b, int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}