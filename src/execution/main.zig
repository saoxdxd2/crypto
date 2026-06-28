const std = @import("std");

const inference = @cImport({
    @cInclude("inference_core.h");
});
const c = @cImport({
    @cInclude("stdio.h");
});

pub fn main() !void {
    // 1. Initialize the Models via C-ABI
    const lobert_path = "../../onnx_exports/lobert.onnx";
    const fincast_path = "../../onnx_exports/fincast.onnx";

    if (inference.init_models(lobert_path, fincast_path) != 0) {
        _ = c.printf("{{\"type\": \"ERROR\", \"msg\": \"Failed to initialize C++ ONNX Engine!\"}}\n");
        _ = c.fflush(null);
        return;
    }
    defer inference.shutdown_models();

    // 2. Safe Memory Allocation using C allocator (since we link libc)
    const allocator = std.heap.c_allocator;

    const expected_size = 640;
    var market_data = try allocator.alloc(f32, expected_size);
    defer allocator.free(market_data);

    // 3. Connect Native TCP Socket to TLS-Termination Proxy
    const proxy_address = try std.net.Address.parseIp4("127.0.0.1", 9000);
    var stream = std.net.tcpConnectToAddress(proxy_address) catch |err| {
         _ = c.printf("{{\"type\": \"ERROR\", \"msg\": \"Zig failed to connect to local TCP proxy: %s\"}}\n", @errorName(err).ptr);
         _ = c.fflush(null);
         return;
    };
    defer stream.close();

    _ = c.printf("{{\"type\": \"STATUS\", \"msg\": \"Zig Native Engine Ready. Connected to TCP Stream.\"}}\n");
    _ = c.fflush(null);

    // 4. The Microsecond Tick Loop (Blocking on Network)
    var reader = stream.reader();
    
    while (true) {
        const bytes_read = reader.readAll(std.mem.sliceAsBytes(market_data)) catch |err| {
            _ = c.printf("{{\"type\": \"ERROR\", \"msg\": \"Zig read error: %s\"}}\n", @errorName(err).ptr);
            break;
        };
        if (bytes_read == 0) break; // EOF
        if (bytes_read != expected_size * @sizeOf(f32)) {
            _ = c.printf("{{\"type\": \"ERROR\", \"msg\": \"Zig partial read\"}}\n");
            continue;
        }

        const timer = try std.time.Timer.start();
        const edge = inference.run_inference(market_data.ptr, @intCast(market_data.len));
        const elapsed = timer.read();
        
        var action: [*c]const u8 = "HOLD";
        var size: u32 = 0;
        if (edge > 0.05) { action = "OPEN"; size = 100; } 
        else if (edge < -0.05) { action = "CLOSE"; size = 100; }
        
        _ = c.printf("{{\"type\": \"SIGNAL\", \"net_edge\": %.6f, \"action\": \"%s\", \"size\": %d, \"reason_code\": \"ZIG_NATIVE_US:%llu\"}}\n", edge, action, size, elapsed);
        _ = c.fflush(null);
    }
}
