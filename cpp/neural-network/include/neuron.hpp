#ifndef NEURON_H
#define NEURON_H

#include <stdio.h>
#include <vector>
#include <cmath>

class Neuron
{
public:
    double output;
    double delta;
    std::vector<double> *weights;
    Neuron(int previous_layer_size, int current_layer_size);
    void initialize_weights(int previous_layer_size);
};

#endif // NEURON_H