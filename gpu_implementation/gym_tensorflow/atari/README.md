Notice
-----------------
The ALE/atari-py is not part of deep-neuroevolution.
This folder provides the instructions and sample code if you are interested in running the ALE.
It depends on atari-py. atari-py is licensed under GPLv2.

Instructions
-----------------

The first thing to do is clone the atari-py repository into the `gym_tensorflow` folder using
```
git clone https://github.com/fps7806/atari-py.git
```
The relative path is important but can be changed inside the `Makefile` as necessary.

We will be using slightly different settings for the build, so you need to go to ./atari-py/atari_py/ale_interface/CMakeLists.txt file and change the first lines to:

```
cmake_minimum_required (VERSION 2.6)
project(ale)
set(ALEVERSION "0.5")


option(USE_SDL "Use SDL" OFF)
option(USE_RLGLUE "Use RL-Glue" OFF)
option(BUILD_EXAMPLES "Build Example Agents" OFF)
option(BUILD_CPP_LIB "Build C++ Shared Library" ON)
option(BUILD_CLI "Build ALE Command Line Interface" OFF)
option(BUILD_C_LIB "Build ALE C Library (needed for Python interface)" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wunused -fPIC -O3 -fomit-frame-pointer -D__STDC_CONSTANT_MACROS -D_GLIBCXX_USE_CXX11_ABI=0")
```

This will ensure that the C++ lib is compiled as well as adding `-D_GLIBCXX_USE_CXX11_ABI=0` which is required for compatibility with TensorFlow.
Once modified you can build the library with `cd ./atari-py && make`.

Once built successfully, the `USE_ALE := 1` flag can be set on the ./gym_tensorflow/Makefile so that the necessary files are compiled.

Building `cd ./gym_tensorflow && make` should give you access to the Atari games as a set of TensorFlow ops.