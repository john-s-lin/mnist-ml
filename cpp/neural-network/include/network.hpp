#ifndef NETWORK_H
#define NETWORK_H

#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "common.hpp"

class Network : public CommonData
{
public:
    std::vector<Layer *> layers;
    double learning_rate;
    double test_performance;
    Network(std::vector<int> spec, int input_size, int output_size, double learning_rate);
    ~Network();

    std::vector<double> feed_forward(Data *data);
    double activation_function(std::vector<double> inputs, std::vector<double> weights);
    double transfer(double activation);
    double transfer_derivative(double output);

    void back_propagate(Data *data);
    void update_weights(Data *data);
    int predict(Data *data);
    void train(int epochs);
    double test();
    void validate();
};

#endif // NETWORK_H