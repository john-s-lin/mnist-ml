# Create a Makefile that will compile the k-nearest-neighbors program.
CC=clang++
CFLAGS := -std=c++17 -Wall -Wextra -pedantic -g 
MNIST_ML_CPP_KNN_ROOT := $(shell pwd)
MNIST_ML_CPP_ROOT := $(shell cd .. && pwd)
SRC := $(MNIST_ML_CPP_KNN_ROOT)/src
INCLUDE_DIR = $(MNIST_ML_CPP_KNN_ROOT)/include
LIB_DIR := $(MNIST_ML_CPP_ROOT)/lib
LIBRARY := libdata.so
DATAHANDLER_SRC = $(MNIST_ML_CPP_ROOT)/src/data_handler.cc

all: euclidean

euclidean: $(SRC)/knn.cc $(DATAHANDLER_SRC)
	$(CC) $(CFLAGS) --define-macro=EUCLIDEAN $(SRC)/knn.cc $(DATAHANDLER_SRC) -o main -L$(LIB_DIR) \
	-I$(INCLUDE_DIR) -I$(MNIST_ML_CPP_ROOT)/include/ -ldata

manhattan: $(SRC)/knn.cc $(DATAHANDLER_SRC)
	$(CC) $(CFLAGS) --define-macro=MANHATTAN $(SRC)/knn.cc $(DATAHANDLER_SRC) -o main -L$(LIB_DIR) \
	-I$(INCLUDE_DIR) -I$(MNIST_ML_CPP_ROOT)/include/ -ldata

clean:
	rm -dfr main*