CC=clang++
CFLAGS := -std=c++17 -Wall -Wextra -pedantic -g 
MNIST_ML_CPP_NN_ROOT := $(shell pwd)
MNIST_ML_CPP_ROOT := $(shell cd .. && pwd)
SRC := $(MNIST_ML_CPP_NN_ROOT)/src
INCLUDE_DIR = $(MNIST_ML_CPP_NN_ROOT)/include
LIB_DIR := $(MNIST_ML_CPP_ROOT)/lib
LIBRARY := libdata.so
DATAHANDLER_SRC = $(MNIST_ML_CPP_ROOT)/src/data_handler.cc

all: main

main: obj $(SRC)/network.cc obj/layer.o obj/neuron.o $(DATAHANDLER_SRC)
	$(CC) $(CFLAGS) $(SRC)/network.cc $(DATAHANDLER_SRC) -o main -L$(LIB_DIR) -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_CPP_ROOT)/include -ldata ./obj/*.o

obj:
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/layer.o: $(SRC)/layer.cc
	$(CC) $(CFLAGS) -c $(SRC)/layer.cc -o obj/layer.o -I$(INCLUDE_DIR) -I$(MNIST_ML_CPP_ROOT)/include/

obj/neuron.o: $(SRC)/neuron.cc
	$(CC) $(CFLAGS) -c $(SRC)/neuron.cc -o obj/neuron.o -I$(INCLUDE_DIR) -I$(MNIST_ML_CPP_ROOT)/include/

clean:
	rm -dfr main*
	rm -dfr obj