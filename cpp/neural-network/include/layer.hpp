#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "neuron.hpp"

class Layer
{
public:
    int current_layer_size;
    std::vector<Neuron *> neurons;
    std::vector<double> layer_output;

    Layer(int current_layer_size, int previous_layer_size);
    ~Layer();
    std::vector<double> get_layer_output();
    int get_layer_size();
};

#endif // LAYER_H