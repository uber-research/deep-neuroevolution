USE_SDL := 0
USE_ALE := 0
USE_GPU := 1

DIR := ./

TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
FLAGS := -std=c++11 -shared -fPIC -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -D_GLIBCXX_USE_CXX11_ABI=0 -O2
CXX := g++
LDFLAGS := -ltensorflow_framework

SOURCES := $(DIR)/*.cpp $(DIR)/ops/*.cpp

ifeq ($(USE_GPU), 1)
    FLAGS += -DGOOGLE_CUDA=1
endif

# This will likely need to be changed to suit your installation.
ifeq ($(USE_ALE), 1)
    ALE := $(shell pwd)/atari-py/atari_py/ale_interface
    FLAGS += -I$(ALE)/src -I$(ALE)/src/controllers -I$(ALE)/src/os_dependent -I$(ALE)/src/environment -I$(ALE)/src/external -L$(ALE)/build
    LDFLAGS += -lale
    SOURCES += $(DIR)/atari/*.cpp
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    ifeq ($(USE_ALE),1)
        FLAGS+= -Wl -rpath=$(ALE)/build
    endif
endif

ifeq ($(UNAME_S),Darwin)
    FLAGS += -framework Cocoa
endif

ifeq ($(strip $(USE_SDL)), 1)
    DEFINES += -D__USE_SDL -DSOUND_SUPPORT
    FLAGS += $(shell sdl-config --cflags)
    LDFLAGS += $(shell sdl-config --libs)
endif


all: gym_tensorflow.so

gym_tensorflow.so:
	$(CXX) $(FLAGS) $(SOURCES) $(LDFLAGS) -o gym_tensorflow.so

clean:
	rm -rf gym_tensorflow.so

remake: clean all
