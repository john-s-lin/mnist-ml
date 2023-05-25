#ifndef COMMON_H
#define COMMON_H

#include "data.hpp"
#include <vector>

class CommonData
{
protected:
    std::vector<Data *> *training_data;
    std::vector<Data *> *test_data;
    std::vector<Data *> *validation_data;

public:
    void set_training_data(std::vector<Data *> *training_data);
    void set_test_data(std::vector<Data *> *test_data);
    void set_validation_data(std::vector<Data *> *validation_data);
};

#endif // COMMON_H