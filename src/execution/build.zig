const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "engine",
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    // Add current directory to include path so it can find inference_core.h
    exe.root_module.addIncludePath(b.path("."));
    
    // Link the C++ library compiled by CMake
    exe.root_module.addLibraryPath(b.path("build/Debug"));
    exe.root_module.addLibraryPath(b.path("build/Release"));
    exe.root_module.linkSystemLibrary("inference_core", .{});

    b.installArtifact(exe);
}
