#include <stdio.h>


// Needed for LLVM Offload:
#ifdef __clang__
#include <offload/cuda_runtime.h>
#endif

// CUDA kernel function: Each thread will execute this function
__global__ void helloKernel() {
    // Print from the GPU device. 
    // Only one thread (thread index 0) will print to avoid duplicate messages.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello, World from the GPU!\n");
    }
}

int main() {
    // Launch the kernel with one block and one thread.
    helloKernel<<<1, 1>>>();

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Optionally, also print a message from the host
    printf("Hello, World from the CPU!\n");

    return 0;
}
