"""
Build script for heston_cpp pybind11 extension.

Install with:
    cd cpp_heston
    pip install .

Or for development:
    pip install -e .
"""
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "heston_cpp",
        ["bindings.cpp"],
        include_dirs=["."],
        cxx_std=17,
        extra_compile_args=["-O3", "-ffast-math"],
    ),
]

setup(
    name="heston_cpp",
    version="0.1.0",
    author="",
    description="Heston FFT Option Pricer (C++ with pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
)
