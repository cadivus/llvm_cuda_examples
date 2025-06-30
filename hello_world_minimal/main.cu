#include <stdio.h>

#include <initializer_list>
#ifdef __clang__
#include <offload/cuda/cuda_runtime.h>
#endif

int main() {
    int *A;
    int R = 5, P = 7;
    cudaMalloc(&A, 4);
    cudaMemcpy(A, &P, 4, cudaMemcpyHostToDevice);

    cudaMemcpy(&R, A, 4, cudaMemcpyDeviceToHost);

    // Optionally, also print a message from the host
    printf("Hello, World from the CPU: %i!\n", R);

    return 0;
}
