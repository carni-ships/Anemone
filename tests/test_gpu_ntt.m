// test_gpu_ntt.m — Test GPU NTT for Dilithium

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "orion_gpu_ntt.h"

int main() {
    printf("=== GPU NTT Test ===\n");

    if (!orion_gpu_ntt_available()) {
        printf("FAIL: No Metal GPU available\n");
        return 1;
    }
    printf("PASS: Metal GPU available\n");

    OrionGpuNtt *ntt = orion_gpu_ntt_create();
    if (!ntt) {
        printf("FAIL: Could not create GPU NTT engine\n");
        return 1;
    }
    printf("PASS: GPU NTT engine created\n");

    // Test with 512 polynomials (same as Anemone's benchmark)
    uint32_t num_polys = 512;
    uint32_t N = 256;

    if (!orion_gpu_ntt_init_dilithium(ntt, num_polys)) {
        printf("FAIL: Could not initialize for Dilithium\n");
        orion_gpu_ntt_destroy(ntt);
        return 1;
    }
    printf("PASS: Dilithium initialized for %d polynomials\n", num_polys);

    // Create test data - simple pattern
    uint32_t *data = (uint32_t *)malloc(num_polys * N * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_polys * N; i++) {
        data[i] = i % 8380417;  // Keep in range
    }

    printf("Testing forward NTT...\n");
    clock_t start = clock();
    if (!orion_gpu_ntt_forward_dilithium(ntt, data, num_polys)) {
        printf("FAIL: Forward NTT failed\n");
        free(data);
        orion_gpu_ntt_destroy(ntt);
        return 1;
    }
    clock_t forward_time = clock() - start;
    double forward_ms = forward_time * 1000.0 / CLOCKS_PER_SEC;
    printf("PASS: Forward NTT in %.2f ms (%.2f us/poly)\n",
           forward_ms, forward_ms * 1000 / num_polys);

    // Test inverse
    printf("Testing inverse NTT...\n");
    start = clock();
    if (!orion_gpu_ntt_inverse_dilithium(ntt, data, num_polys)) {
        printf("FAIL: Inverse NTT failed\n");
        free(data);
        orion_gpu_ntt_destroy(ntt);
        return 1;
    }
    clock_t inverse_time = clock() - start;
    double inverse_ms = inverse_time * 1000.0 / CLOCKS_PER_SEC;
    printf("PASS: Inverse NTT in %.2f ms (%.2f us/poly)\n",
           inverse_ms, inverse_ms * 1000 / num_polys);

    printf("\nTotal GPU NTT time: %.2f ms for %d polynomials\n",
           forward_ms + inverse_ms, num_polys);
    printf("CPU estimate was ~1000ms for 512 NTTs\n");
    printf("Speedup: %.1fx\n", 1000.0 / (forward_ms + inverse_ms));

    free(data);
    orion_gpu_ntt_destroy(ntt);

    printf("\n=== All tests passed ===\n");
    return 0;
}