#include "common.hpp"

void CommonData::set_training_data(std::vector<Data *> *training_data)
{
    this->training_data = training_data;
}

void CommonData::set_test_data(std::vector<Data *> *test_data)
{
    this->test_data = test_data;
}

void CommonData::set_validation_data(std::vector<Data *> *validation_data)
{
    this->validation_data = validation_data;
}
