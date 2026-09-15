DEEPSTREAM_ROOT ?= /opt/nvidia/deepstream/deepstream-6.2
CXX ?= g++
CXXFLAGS ?= -O3
CXXFLAGS += -std=c++14 -Wall -Wextra -fPIC \
	-I$(DEEPSTREAM_ROOT)/sources/includes \
	-I/usr/local/cuda/include
LDFLAGS += -shared

TARGET := libnvdsinfer_custom_yolov5.so
SOURCE := custom_yolov5_parser.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) $(LDFLAGS) -o $@

clean:
	$(RM) $(TARGET)
