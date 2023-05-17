#ifndef DATA_H
#define DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"

class Data
{
    std::vector<uint8_t> *feature_vector;
    uint8_t label;
    int enum_label;

public:
    Data();
    ~Data();
    void set_feature_vector(std::vector<uint8_t> *feature_vector);
    void append_feature_vector(uint8_t feature);
    void set_label(uint8_t label);
    void set_enum_label(int enum_label);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enum_label();

    std::vector<uint8_t> *get_feature_vector();
};

#endif // DATA_H