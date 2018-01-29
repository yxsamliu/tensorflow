# Description:
#   Wrap AMD (https://github.com/ROCmSoftwarePlatform/rccl) RCCL with tensorflow ops.
#   APIs are meant to change over time.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "tf_gpu_cc_test")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_rccl_ops.so",
    srcs = [
        "kernels/rccl_manager.cc",
        "kernels/rccl_manager.h",
        "kernels/rccl_ops.cc",
        "ops/rccl_ops.cc",
    ],
    deps = [
        "//tensorflow/core:gpu_headers_lib",
        "@rccl_archive//:rccl",
    ],
)

tf_gpu_cc_test(
    name = "rccl_manager_test",
    size = "medium",
    srcs = if_rocm(
        [
            "kernels/rccl_manager.cc",
            "kernels/rccl_manager.h",
            "kernels/rccl_manager_test.cc",
        ],
        [],
    ),
    # Disabled on jenkins until errors finding nvmlShutdown are found.
    tags = [
        "manual",
        "notap",
    ],
    deps = if_rocm(
        [
            "@rccl_archive//:rccl",
            "//tensorflow/core:rocm",
        ],
        [],
    ) + [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "rccl_kernels",
    srcs = [
        "kernels/rccl_manager.cc",
        "kernels/rccl_manager.h",
        "kernels/rccl_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "@rccl_archive//:rccl",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["rccl_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "rccl_ops",
    deps = [":rccl_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "rccl_py",
    srcs = [
        "__init__.py",
        "python/ops/rccl_ops.py",
    ],
    dso = [
        ":python/ops/_rccl_ops.so",
    ],
    kernels = [
        ":rccl_kernels",
        ":rccl_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":rccl_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
    ],
)

# http://b/62064807
# rocm_py_test(
#     name = "rccl_ops_test",
#     size = "small",
#     srcs = ["python/ops/rccl_ops_test.py"],
#     additional_deps = [
#         ":rccl_py",
#         "//tensorflow/python:array_ops",
#         "//tensorflow/python:client_testlib",
#         "//tensorflow/python:framework_for_generated_wrappers",
#         "//tensorflow/python:framework_test_lib",
#         "//tensorflow/python:platform_test",
#     ],
#     # Disabled on jenkins until errors finding nvmlShutdown are found.
#     tags = [
#         "manual",
#         "notap",
#     ],
# )

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
