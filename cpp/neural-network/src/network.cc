#include "network.hpp"
#include "layer.hpp"
#include "data_handler.hpp"
#include <numeric>

Network::Network(std::vector<int> layer_sizes, int input_size, int output_size, double learning_rate)
{
    for (int i = 0; i < layer_sizes.size(); i++)
    {
        if (i == 0)
        {
            layers.push_back(new Layer(input_size, layer_sizes.at(i)));
        }
        else
        {
            layers.push_back(new Layer(layers.at(i - 1)->neurons.size(), layer_sizes.at(i)));
        }
    }
    layers.push_back(new Layer(layers.at(layers.size() - 1)->neurons.size(), output_size));
    this->learning_rate = learning_rate;
}

Network::~Network()
{
    for (Layer *layer : layers)
    {
        delete layer;
    }
}

/**
 * @brief Forward propagate the input data through the network
 *
 * @param data
 * @return std::vector<double>
 */
std::vector<double> Network::feed_forward(Data *data)
{
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); i++)
    {
        Layer *layer = layers.at(i);
        std::vector<double> new_inputs;
        for (Neuron *n : layer->neurons)
        {
            double activation = this->activation_function(inputs, *n->weights);
            n->output = this->transfer(activation);
            new_inputs.push_back(n->output);
        }
        inputs = new_inputs;
    }
    return inputs;
}

double Network::activation_function(std::vector<double> inputs, std::vector<double> weights)
{
    double activation = weights.back();
    for (int i = 0; i < weights.size() - 1; i++)
    {
        activation += weights[i] * inputs[i];
    }
    return activation;
}

/**
 * @brief Sigmoid activation function
 *
 * @param activation
 * @return double
 */
double Network::transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation));
}

/**
 * @brief Derivative of the sigmoid function
 *
 * @param output
 * @return double
 */
double Network::transfer_derivative(double output)
{
    return output * (1.0 - output);
}

/**
 * @brief Update the weights of the network
 *
 * @param data
 */
void Network::back_propagate(Data *data)
{
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        Layer *layer = layers.at(i);
        std::vector<double> errors;
        if (i != layers.size() - 1)
        {
            for (int j = 0; j < layer->neurons.size(); j++)
            {
                double error = 0;
                for (Neuron *n : layers.at(i + 1)->neurons)
                {
                    error += (n->weights->at(j) * n->delta);
                }
                errors.push_back(error);
            }
        }
        else
        {
            for (int j = 0; j < layer->neurons.size(); j++)
            {
                Neuron *n = layer->neurons.at(j);
                errors.push_back((double)data->get_class_vector()->at(j) - n->output);
            }
        }
        for (int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron *n = layer->neurons.at(j);
            n->delta = errors.at(j) * this->transfer_derivative(n->output);
        }
    }
}

/**
 * @brief Update the weights of the network
 *
 * @param data
 */
void Network::update_weights(Data *data)
{
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); i++)
    {
        if (i != 0)
        {
            for (Neuron *n : layers.at(i - 1)->neurons)
            {
                inputs.push_back(n->output);
            }
        }
        for (Neuron *n : layers.at(i)->neurons)
        {
            for (int j = 0; j < inputs.size(); j++)
            {
                n->weights->at(j) += this->learning_rate * n->delta * inputs.at(j);
            }
            n->weights->back() += this->learning_rate * n->delta;
        }
        inputs.clear();
    }
}

/**
 * @brief Predict the class of the input data
 *
 * @param data
 * @return int
 */
int Network::predict(Data *data)
{
    std::vector<double> outputs = this->feed_forward(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

/**
 * @brief Train the network
 *
 * @param epochs
 */
void Network::train(int epochs)
{
    for (int i = 0; i < epochs; i++)
    {
        double sum_error = 0;
        for (Data *data : *this->training_data)
        {
            std::vector<double> outputs = this->feed_forward(data);
            std::vector<int> expected = *data->get_class_vector();
            sum_error += std::inner_product(outputs.begin(), outputs.end(), expected.begin(), 0.0, std::plus<>(), [](double a, double b)
                                            { return pow(a - b, 2); });
            this->back_propagate(data);
            this->update_weights(data);
        }
        printf("Epoch: %d, Error: %.4f\n", i, sum_error);
    }
}

/**
 * @brief Test the network
 *
 */
double Network::test()
{
    int correct = 0;
    for (Data *data : *this->test_data)
    {
        int prediction = this->predict(data);
        int actual = std::distance(data->get_class_vector()->begin(), std::max_element(data->get_class_vector()->begin(), data->get_class_vector()->end()));
        if (prediction == actual)
        {
            correct++;
        }
    }
    return (double)correct / (double)this->test_data->size();
}

/**
 * @brief Validate the network
 *
 */
void Network::validate()
{
    int correct = 0;
    for (Data *data : *this->validation_data)
    {
        int prediction = this->predict(data);
        int actual = std::distance(data->get_class_vector()->begin(), std::max_element(data->get_class_vector()->begin(), data->get_class_vector()->end()));
        if (prediction == actual)
        {
            correct++;
        }
    }
    printf("Validation accuracy: %.4f\n", (double)correct / (double)this->validation_data->size());
}

int main()
{
    DataHandler *dh = new DataHandler();
#ifdef MNIST
    dh->read_feature_vector("data/train-images-idx3-ubyte");
    dh->read_class_vector("data/train-labels-idx1-ubyte");
    dh->count_classes();
#else
    dh->read_csv("data/iris.csv", ",");
#endif
    dh->split_data();
    std::vector<int> hidden_layers = {10};
    auto lambda = [&]()
    {
        Network *nn = new Network(
            hidden_layers,
            dh->get_training_data()->at(0)->get_normalized_feature_vector()->size(),
            dh->get_num_classes(),
            0.25);
        nn->set_training_data(dh->get_training_data());
        nn->set_test_data(dh->get_test_data());
        nn->set_validation_data(dh->get_validation_data());
        nn->train(15);
        nn->validate();
        printf("Test accuracy: %.4f\n", nn->test());
    };
    lambda();
}