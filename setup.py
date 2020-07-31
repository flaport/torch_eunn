#   Copyright 2018 Floris Laporte
#   MIT License

#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the Software
#   without restriction, including without limitation the rights to use, copy, modify,
#   merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following
#   conditions:

#   The above copyright notice and this permission notice shall be included in all copies
#   or substantial portions of the Software.

#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
#   THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch_eunn
from setuptools import setup, Extension
from torch.utils import cpp_extension

with open("readme.md", "r") as f:
    long_description = f.read()

torch_eunn_cpp = Extension(
    name="torch_eunn_cpp",
    sources=["torch_eunn.cpp"],
    include_dirs=cpp_extension.include_paths(),
    library_dirs=cpp_extension.library_paths(),
    extra_compile_args=[],
    libraries=[
        "c10",
        "torch",
        "torch_cpu",
        "torch_python",
    ],
    language="c++",
)

setup(
    name="torch_eunn",
    version=torch_eunn.__version__,
    author=torch_eunn.__author__,
    author_email="floris.laporte@gmail.com",
    description=torch_eunn.__doc__.splitlines()[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flaport/torch_eunn",
    py_modules=["torch_eunn"],
    ext_modules=[torch_eunn_cpp],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)

