# Description: rocPRIM library which is a set of primitives for GPU programming on AMD ROCm stack.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

load("@local_config_rocm//rocm:build_defs.bzl", "rocm_default_copts", "if_rocm")

filegroup(
    name = "hipcub_header_files",
    srcs = glob([
        "hipcub/**",
    ]),
)

cc_library(
    name = "hipcub",
    hdrs = if_rocm([":hipcub_header_files"]),
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
)
