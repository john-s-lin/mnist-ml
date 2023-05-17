#include "data.hpp"

/**
 * @brief Construct a new Data:: Data object
 *
 */
Data::Data()
{
    this->feature_vector = new std::vector<uint8_t>();
}

/**
 * @brief Destroy the Data:: Data object
 *
 */
Data::~Data()
{
    delete this->feature_vector;
}

/**
 * @brief Set the feature vector object
 *
 * @param feature_vector
 */
void Data::set_feature_vector(std::vector<uint8_t> *feature_vector)
{
    this->feature_vector = feature_vector;
}

/**
 * @brief Append a feature to the feature vector
 *
 * @param feature
 */
void Data::append_feature_vector(uint8_t feature)
{
    this->feature_vector->push_back(feature);
}

/**
 * @brief Set the label object
 *
 * @param label
 */
void Data::set_label(uint8_t label)
{
    this->label = label;
}

/**
 * @brief Set the enum label object
 *
 * @param enum_label
 */
void Data::set_enum_label(int enum_label)
{
    this->enum_label = enum_label;
}

/**
 * @brief Get the feature vector size
 *
 * @return int
 */
int Data::get_feature_vector_size()
{
    return this->feature_vector->size();
}

/**
 * @brief Get the label
 *
 * @return uint8_t
 */
uint8_t Data::get_label()
{
    return this->label;
}

/**
 * @brief Get the enum label
 *
 * @return uint8_t
 */
uint8_t Data::get_enum_label()
{
    return this->enum_label;
}

/**
 * @brief Get the feature vector
 *
 * @return std::vector<uint8_t>*
 */
std::vector<uint8_t> *Data::get_feature_vector()
{
    return this->feature_vector;
}