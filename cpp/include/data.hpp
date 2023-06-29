#ifndef DATA_H
#define DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"

class Data
{
    std::vector<uint8_t> *feature_vector;
    std::vector<double> *normalized_feature_vector;
    std::vector<int> *class_vector;
    uint8_t label;
    int enum_label;
    double distance;

public:
    Data();
    ~Data();
    void set_feature_vector(std::vector<uint8_t> *feature_vector);
    void set_normalized_feature_vector(std::vector<double> *double_feature_vector);

    void append_feature_vector(uint8_t feature);
    void append_feature_vector(double feature);
    void set_class_vector(int count);
    void set_label(uint8_t label);
    void set_enum_label(int enum_label);
    void set_distance(double distance);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enum_label();

    std::vector<uint8_t> *get_feature_vector();
    std::vector<double> *get_normalized_feature_vector();
    std::vector<int> *get_class_vector();
};

#endif // DATA_H