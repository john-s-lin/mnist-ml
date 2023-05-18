#include "data_handler.hpp"

/**
 * @brief Construct a new DataHandler:: DataHandler object
 *
 */
DataHandler::DataHandler()
{
    data_array = new std::vector<Data *>();
    training_data = new std::vector<Data *>();
    test_data = new std::vector<Data *>();
    validation_data = new std::vector<Data *>();
}

/**
 * @brief Destroy the DataHandler:: DataHandler object
 *
 */
DataHandler::~DataHandler()
{
    delete data_array;
    delete training_data;
    delete test_data;
    delete validation_data;
}

/**
 * @brief Read the feature vector from the file
 *
 * @param path
 */
void DataHandler::read_feature_vector(std::string path)
{
    uint32_t header[4]; // |magic|num_images|num_rows|num_cols|
    unsigned char bytes[4];

    FILE *file = fopen(path.c_str(), "r");
    if (file)
    {
        for (int i = 0; i < 4; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, file))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done reading input file header.\n");
        int image_size = header[2] * header[3];
        for (size_t i = 0; i < header[1]; i++)
        {
            Data *data = new Data();
            uint8_t element[1];
            for (int j = 0; j < image_size; j++)
            {
                if (fread(element, sizeof(element), 1, file))
                {
                    data->append_feature_vector(element[0]);
                }
                else
                {
                    printf("Error reading file.\n");
                    exit(1);
                }
            }
            data_array->push_back(data);
        }
        printf("Done reading %lu feature vectors.\n", data_array->size());
    }
    else
    {
        printf("Error opening file.\n");
        exit(1);
    }
}

/**
 * @brief Read the feature labels from the file
 *
 * @param path
 */
void DataHandler::read_feature_labels(std::string path)
{
    uint32_t header[2]; // |magic|num_images|
    unsigned char bytes[4];

    FILE *file = fopen(path.c_str(), "r");
    if (file)
    {
        for (int i = 0; i < 2; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, file))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done reading label file header.\n");
        for (size_t i = 0; i < header[1]; i++)
        {
            uint8_t element[1];

            if (fread(element, sizeof(element), 1, file))
            {
                data_array->at(i)->set_label(element[0]);
            }
            else
            {
                printf("Error reading file.\n");
                exit(1);
            }
        }
        printf("Done reading label.\n");
    }
    else
    {
        printf("Error opening file.\n");
        exit(1);
    }
}

/**
 * @brief Split the data into training, test, and validation sets
 *
 */
void DataHandler::split_data()
{
    std::unordered_set<int> used_indexes;
    size_t train_size = data_array->size() * TRAIN_SET_PERCENT;
    size_t test_size = data_array->size() * TEST_SET_PERCENT;
    size_t validation_size = data_array->size() * VALIDATION_SET_PERCENT;

    // Training set
    while (training_data->size() < train_size)
    {
        int index = rand() % data_array->size();
        if (used_indexes.find(index) == used_indexes.end())
        {
            training_data->push_back(data_array->at(index));
            used_indexes.insert(index);
        }
    }

    // Test set
    while (test_data->size() < test_size)
    {
        int index = rand() % data_array->size();
        if (used_indexes.find(index) == used_indexes.end())
        {
            test_data->push_back(data_array->at(index));
            used_indexes.insert(index);
        }
    }

    // Validation set
    while (validation_data->size() < validation_size)
    {
        int index = rand() % data_array->size();
        if (used_indexes.find(index) == used_indexes.end())
        {
            validation_data->push_back(data_array->at(index));
            used_indexes.insert(index);
        }
    }

    printf("Training set size: %lu\n", training_data->size());
    printf("Test set size: %lu\n", test_data->size());
    printf("Validation set size: %lu\n", validation_data->size());
}

/**
 * @brief Count the number of classes in the data
 *
 */
void DataHandler::count_classes()
{
    int count = 0;
    for (unsigned i = 0; i < data_array->size(); i++)
    {
        if (class_labels.find(data_array->at(i)->get_label()) == class_labels.end())
        {
            class_labels[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enum_label(count);
            count++;
        }
    }
    num_classes = count;
    printf("Number of classes: %d\n", num_classes);
}

/**
 * @brief Convert the bytes to little endian
 *
 * @param bytes
 * @return uint32_t
 */
uint32_t DataHandler::convert_to_little_endian(const unsigned char *bytes)
{
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

/**
 * @brief Get the training data object
 *
 * @return std::vector<Data *>*
 */
std::vector<Data *> *DataHandler::get_training_data()
{
    return training_data;
}

/**
 * @brief Get the test data object
 *
 * @return std::vector<Data *>*
 */
std::vector<Data *> *DataHandler::get_test_data()
{
    return test_data;
}

/**
 * @brief Get the validation data object
 *
 * @return std::vector<Data *>*
 */
std::vector<Data *> *DataHandler::get_validation_data()
{
    return validation_data;
}

// int main()
// {
//     DataHandler *data_handler = new DataHandler();
//     data_handler->read_feature_vector("../data/train-images-idx3-ubyte");
//     data_handler->read_feature_labels("../data/train-labels-idx1-ubyte");
//     data_handler->split_data();
//     data_handler->count_classes();
// }