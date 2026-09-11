import platform
from setuptools import setup, Extension
import pybind11

if platform.system() == "Windows":
    extra_compile_args = ["/O2", "/std:c++17", "/EHsc", "/DNOMINMAX"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-std=c++17", "-ffast-math", "-fvisibility=hidden"]
    extra_link_args = ["-pthread"]

setup(
    name='diffusion_model',
    version='1.0',
    ext_modules=[
        Extension(
            'diffusion_model',
            sources=['socialsis_wrapper.cpp'],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language='c++',
        ),
    ],
)