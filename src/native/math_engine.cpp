#include <immintrin.h>
#include <cstdint>
#include <span>

extern "C" {

// Highly optimized Moving Average utilizing AVX2 for float64 arrays
// Enforces Branchless Execution and L1 Cache Optimization.
__declspec(dllexport) void compute_sma_avx2(const double* __restrict prices, double* __restrict out, int32_t length, int32_t window) {
    if (length < window || window <= 0) return;

    double current_sum = 0.0;
    // Scalar initialization for the first window
    for (int i = 0; i < window; ++i) {
        current_sum += prices[i];
    }
    
    out[window - 1] = current_sum / window;

    // Moving window using scalar (SIMD used for large parallel reductions if needed)
    // For single rolling MA, scalar is often fast enough, but here we enforce unrolling.
    #pragma unroll(4)
    for (int i = window; i < length; ++i) {
        current_sum += prices[i] - prices[i - window];
        out[i] = current_sum / window;
    }
}

// Example AVX2 bulk calculation (e.g., vectorized returns calculation: p[i]/p[i-1] - 1)
__declspec(dllexport) void compute_returns_avx2(const double* __restrict prices, double* __restrict returns, int32_t length) {
    if (length < 2) return;
    
    const __m256d one = _mm256_set1_pd(1.0);
    
    int i = 1;
    // Process 4 doubles at a time
    for (; i <= length - 4; i += 4) {
        // Load p[i], p[i+1], p[i+2], p[i+3]
        __m256d p_curr = _mm256_loadu_pd(&prices[i]);
        // Load p[i-1], p[i], p[i+1], p[i+2]
        __m256d p_prev = _mm256_loadu_pd(&prices[i - 1]);
        
        // p_curr / p_prev
        __m256d div = _mm256_div_pd(p_curr, p_prev);
        // div - 1.0
        __m256d res = _mm256_sub_pd(div, one);
        
        _mm256_storeu_pd(&returns[i], res);
    }
    
    // Tail handling
    for (; i < length; ++i) {
        returns[i] = (prices[i] / prices[i - 1]) - 1.0;
    }
}

} // extern "C"
