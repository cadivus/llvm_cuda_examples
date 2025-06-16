#include <stdio.h>

#include <initializer_list>
#ifdef __clang__
#include <offload/cuda/cuda_runtime.h>
#endif

// CUDA kernel function: Each thread will execute this function
__global__ void helloKernel(int *A) {
    // Print from the GPU device.
    // Only one thread (thread index 0) will print to avoid duplicate messages.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
	      *A = 42;
	      // Broken
        // printf("Hello, World from the GPU %i!\n", 0);
    }
}

int main() {
    int *A;
    int R = 5, P = 7;
    cudaMalloc(&A, 4);
    cudaMemcpy(A, &P, 4, cudaMemcpyHostToDevice);
    // Launch the kernel with one block and one thread.
    helloKernel<<<1, 1>>>(A);

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaMemcpy(&R, A, 4, cudaMemcpyDeviceToHost);

    // Optionally, also print a message from the host
    printf("Hello, World from the CPU: %i!\n", R);

    return 0;
}
