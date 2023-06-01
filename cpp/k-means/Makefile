# Create a Makefile that will compile the k-means program.
CC=clang++
CFLAGS := -std=c++17 -Wall -Wextra -pedantic -g 
MNIST_ML_CPP_KMEANS_ROOT := $(shell pwd)
MNIST_ML_CPP_ROOT := $(shell cd .. && pwd)
SRC := $(MNIST_ML_CPP_KMEANS_ROOT)/src
INCLUDE_DIR = $(MNIST_ML_CPP_KMEANS_ROOT)/include
LIB_DIR := $(MNIST_ML_CPP_ROOT)/lib
LIBRARY := libdata.so
DATAHANDLER_SRC = $(MNIST_ML_CPP_ROOT)/src/data_handler.cc

all: main

main: $(SRC)/k_means.cc $(DATAHANDLER_SRC)
	$(CC) $(CFLAGS) --define-macro=EUCLIDEAN $(SRC)/k_means.cc $(DATAHANDLER_SRC) -o kmeans -L$(LIB_DIR) \
	-I$(INCLUDE_DIR) -I$(MNIST_ML_CPP_ROOT)/include/ -ldata

clean:
	rm -dfr kmeans*