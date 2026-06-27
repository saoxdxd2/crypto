const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const lib = b.addSharedLibrary(.{
        .name = "math_engine_zig",
        .root_source_file = .{ .path = "math_engine.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Force AVX2 or similar target features if we know the architecture
    // Usually handled via standardTargetOptions, e.g. `zig build -Dtarget=x86_64-windows -Dcpu=haswell`
    lib.linkLibC();
    
    // Output directly to crypto_research for python ctypes/cffi access
    const install_step = b.addInstallArtifact(lib, .{
        .dest_dir = .{ .custom = "../../crypto_research" },
    });
    b.getInstallStep().dependOn(&install_step.step);
}
