#include "layer.hpp"

Layer::Layer(int previous_layer_size, int current_layer_size)
{
    for (int i = 0; i < current_layer_size; i++)
    {
        neurons.push_back(new Neuron(previous_layer_size, current_layer_size));
    }
    this->current_layer_size = current_layer_size;
}

Layer::~Layer()
{
    for (int i = 0; i < current_layer_size; i++)
    {
        delete neurons.at(i);
    }
}

std::vector<double> Layer::get_layer_output()
{
    return layer_output;
}

int Layer::get_layer_size()
{
    return this->current_layer_size;
}