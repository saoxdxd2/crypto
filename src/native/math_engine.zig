const std = @import("std");

// Exported function to be called from Python via ctypes or cffi.
// Zig is highly optimized, small, and provides safety with C ABI compatibility.
export fn compute_ewma_zig(
    prices: [*]const f64,
    out: [*]f64,
    length: i32,
    alpha: f64,
) void {
    if (length <= 0) return;

    var current_ewma: f64 = prices[0];
    out[0] = current_ewma;

    // Small, branchless-friendly fast loop
    var i: usize = 1;
    const len: usize = @intCast(length);
    while (i < len) : (i += 1) {
        current_ewma = (prices[i] * alpha) + (current_ewma * (1.0 - alpha));
        out[i] = current_ewma;
    }
}
