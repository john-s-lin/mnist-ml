#ifndef KNN_H
#define KNN_H

#include <vector>
#include "data.hpp"

class KNN
{
    int k;
    std::vector<Data *> *neighbours;

public:
    KNN(int k);
    KNN();
    ~KNN();

    void find_knearest(Data *query_point);
    void set_k(int k);

    int predict();
    double calculate_distance(Data *query_point, Data *input);
    double validate_performance();
    double test_performance();
};

#endif // KNN_H