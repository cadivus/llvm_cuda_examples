#include <stdio.h>
#include <omp.h>

int main(void) {
    int x = 1;

    printf("OpenMP reports %d device(s)\n", omp_get_num_devices());
    printf("Default device = %d\n", omp_get_default_device());

    // Offload a tiny computation to the GPU; bring the result back to host.
    #pragma omp target map(tofrom: x)
    {
        // If this prints 0, we were on a device. If 1, it fell back to host.
        printf("[device] omp_is_initial_device() = %d\n", omp_is_initial_device());
        printf("[device] Default device = %d\n", omp_get_default_device());
        x += 41; // simple work on the GPU
    }

    printf("[host] result = %d\n", x); // should be 42 if the offload ran
    return 0;
}
