#include <stdio.h>

#include <initializer_list>
#ifdef __clang__
#include <offload/hip/hip_runtime.h>
#endif

int main() {
    int *A;
    int R = 5, P = 7;

    hipMalloc((void**)&A, sizeof(int));
    hipMemcpy(A, &P, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(&R, A, sizeof(int), hipMemcpyDeviceToHost);

    // Print result
    printf("Hello, World from the CPU: %i!\n", R);

    // Free device memory
    hipFree(A);

    return 0;
}
