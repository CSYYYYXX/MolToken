from setuptools import Extension, setup
import pybind11

include = [pybind11.get_include()]
setup(ext_modules=[
    Extension("_rebuttal_fddfs_cpp",
              ["pipeline/serialization/_stackk_cpp.cpp"],
              define_macros=[("FDDFS_MODULE_NAME", "_rebuttal_fddfs_cpp")],
              include_dirs=include, language="c++",
              extra_compile_args=["-O3", "-std=c++17", "-Wall"]),
    Extension("pipeline.serialization._stackk_cpp",
              ["pipeline/serialization/_stackk_cpp.cpp"],
              include_dirs=include, language="c++",
              extra_compile_args=["-O3", "-std=c++17", "-Wall"]),
    Extension("src.algorithms.compression._cpp_bpe",
              ["src/algorithms/compression/_cpp_bpe.cpp"],
              include_dirs=include, language="c++",
              extra_compile_args=["-O3", "-std=c++17", "-Wall", "-pthread"],
              extra_link_args=["-pthread"]),
])
