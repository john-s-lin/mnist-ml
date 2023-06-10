#include <cmath>
#include <limits>
#include <map>
#include "data_handler.hpp"
#include "../include/knn.hpp"
#include "stdint.h"

/**
 * @brief Construct a new KNN::KNN object
 *
 * @param k
 */
KNN::KNN(int k)
{
    this->k = k;
}

/**
 * @brief Construct a new KNN::KNN object
 *
 */
KNN::KNN()
{
    this->k = 1;
}

/**
 * @brief Destroy the KNN::KNN object
 *
 */
KNN::~KNN()
{
    delete neighbours;
}

/**
 * @brief Find the k nearest neighbours
 *
 * @param query_point
 */
void KNN::find_knearest(Data *query_point)
{
    neighbours = new std::vector<Data *>();
    double min = std::numeric_limits<double>::max();
    double prev_min = min;
    int index = 0;
    for (int i = 0; i < k; i++)
    {
        if (i == 0)
        {
            for (unsigned j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                if (distance < min)
                {
                    min = distance;
                    index = j;
                }
            }
            neighbours->push_back(training_data->at(index));
            prev_min = min;
            min = std::numeric_limits<double>::max();
        }
        else
        {
            for (unsigned j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                if (distance < min && distance > prev_min)
                {
                    min = distance;
                    index = j;
                }
            }
            neighbours->push_back(training_data->at(index));
            prev_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}

/**
 * @brief Set k
 *
 * @param k
 */
void KNN::set_k(int k)
{
    this->k = k;
}

/**
 * @brief Predict the class of the query point
 *
 * @return int
 */
int KNN::predict()
{
    std::map<uint8_t, int> class_count;
    for (unsigned i = 0; i < neighbours->size(); i++)
    {
        if (class_count.find(neighbours->at(i)->get_label()) == class_count.end())
        {
            class_count[neighbours->at(i)->get_label()] = 1;
        }
        else
        {
            class_count[neighbours->at(i)->get_label()]++;
        }
    }

    int best = 0;
    int max = 0;
    for (auto kv : class_count)
    {
        if (kv.second > max)
        {
            max = kv.second;
            best = kv.first;
        }
    }
    return best;
}

/**
 * @brief Calculate the distance between two data points
 *
 * @param query_point
 * @param input
 * @return double
 */
double KNN::calculate_distance(Data *query_point, Data *input)
{
    double distance = 0.0;
    if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        printf("Error: Feature vector sizes do not match\n");
        exit(1);
    }
#ifdef EUCLIDEAN
    for (int i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        // Note: You don't need square root since it's an extra operation
        distance += pow((query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i)), 2);
    }
#elif defined MANHATTAN // #elifdef will be available in C++23
    for (int i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance += abs(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i));
    }
#endif
    return distance;
}

/**
 * @brief Evaluate the performance of the model on the validation set
 *
 * @return double
 */
double KNN::validate_performance()
{
    double current_performance = 0;
    int count = 0;
    int data_index = 0;

    for (Data *query_point : *validation_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        printf("Query point %d: Prediction = %d, Actual = %d\n", data_index, prediction, query_point->get_label());
        if (prediction == query_point->get_label())
        {
            count++;
        }
        data_index++;
        printf("Current performance = %.4f %%\n", (double)count / (double)(data_index)*100.0);
    }
    current_performance = (double)count / (double)(validation_data->size()) * 100.0;
    printf("Validation performance for K = %d: %.4f %%\n", k, current_performance);
    return current_performance;
}

/**
 * @brief Evaluate the performance of the model on the test set
 *
 * @return double
 */
double KNN::test_performance()
{
    double current_performance = 0;
    int count = 0;

    for (Data *query_point : *test_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            count++;
        }
    }
    current_performance = (double)count / (double)(test_data->size()) * 100.0;
    printf("Test performance = %.4f %%\n", current_performance);
    return current_performance;
}

int main()
{
    DataHandler *data_handler = new DataHandler();
    data_handler->read_feature_vector("../../data/train-images-idx3-ubyte");
    data_handler->read_feature_labels("../../data/train-labels-idx1-ubyte");
    data_handler->split_data();
    data_handler->count_classes();

    KNN *knn = new KNN();
    knn->set_training_data(data_handler->get_training_data());
    knn->set_test_data(data_handler->get_test_data());
    knn->set_validation_data(data_handler->get_validation_data());

    double performance = 0;
    double best_performance = 0;
    int best_k = 1;

    for (int k = 1; k <= 10; k++)
    {
        knn->set_k(k);
        performance = knn->validate_performance();
        if (performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }
    }

    printf("Best K = %d\n", best_k);
    knn->set_k(best_k);
    knn->test_performance();
}