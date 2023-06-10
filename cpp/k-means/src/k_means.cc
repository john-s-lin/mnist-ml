#include "../include/k_means.hpp"

/**
 * @brief Construct a new KMeans::KMeans object
 *
 * @param k
 */
KMeans::KMeans(int k) : CommonData()
{
    this->k = k;
    clusters = new std::vector<cluster_t *>();
    used_points = new std::unordered_set<int>();
}

/**
 * @brief Destroy the KMeans::KMeans object
 *
 */
KMeans::~KMeans()
{
    for (auto cluster : *clusters)
    {
        delete cluster;
    }
    delete clusters;
    delete used_points;
}

/**
 * @brief Initialize the clusters
 *
 */
void KMeans::init_clusters()
{
    for (int i = 0; i < k; i++)
    {
        int centroid_index = rand() % training_data->size();
        while (used_points->find(centroid_index) != used_points->end())
        {
            centroid_index = rand() % training_data->size();
        }
        used_points->insert(centroid_index);
        clusters->push_back(new cluster_t(training_data->at(centroid_index)));
    }
}

/**
 * @brief Initialize the clusters for each class
 */
void KMeans::init_clusters_for_class()
{
    std::unordered_set<int> used_centroids;
    for (uint32_t i = 0; i < training_data->size(); i++)
    {
        if (used_centroids.find(training_data->at(i)->get_label()) == used_centroids.end())
        {
            clusters->push_back(new cluster_t(training_data->at(i)));
            used_centroids.insert(training_data->at(i)->get_label());
            used_points->insert(i);
        }
    }
}

/**
 * @brief Train the model
 *
 */
void KMeans::train()
{
    while (used_points->size() < training_data->size())
    {
        int index = rand() % training_data->size();
        while (used_points->find(index) != used_points->end())
        {
            index = rand() % training_data->size();
        }
        double min_distance = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (uint32_t i = 0; i < clusters->size(); i++)
        {
            double distance = euclidean_distance(clusters->at(i)->centroid, training_data->at(index));
            if (distance < min_distance)
            {
                min_distance = distance;
                best_cluster = i;
            }
        }
        clusters->at(best_cluster)->add_to_cluster(training_data->at(index));
        used_points->insert(index);
        // printf("Assigning point %d to cluster %d\n", index, best_cluster);
    }
}

/**
 * @brief Calculate the euclidean distance between a centroid and a point
 *
 * @param centroid
 * @param point
 * @return double
 */
double KMeans::euclidean_distance(std::vector<double> *centroid, Data *point)
{
    double distance = 0;
    for (uint32_t i = 0; i < centroid->size(); i++)
    {
        distance += pow(centroid->at(i) - point->get_feature_vector()->at(i), 2);
    }
    // Just return squared distance, no need to sqrt since the order is preserved
    return distance;
}

/**
 * @brief Validate the model
 *
 * @return double
 */
double KMeans::validate()
{
    double correct = 0;
    for (auto query_point : *validation_data)
    {
        double min_distance = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (uint32_t i = 0; i < clusters->size(); i++)
        {
            double distance = euclidean_distance(clusters->at(i)->centroid, query_point);
            if (distance < min_distance)
            {
                min_distance = distance;
                best_cluster = i;
            }
        }
        if (clusters->at(best_cluster)->most_common_class == query_point->get_label())
        {
            correct++;
        }
        // printf("Point %d assigned to cluster %d\n", query_point->get_label(), best_cluster);
    }
    return (correct / validation_data->size()) * 100;
}

/**
 * @brief Test the model
 *
 * @return double
 */
double KMeans::test()
{
    double correct = 0;
    for (auto query_point : *test_data)
    {
        double min_distance = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (uint32_t i = 0; i < clusters->size(); i++)
        {
            double distance = euclidean_distance(clusters->at(i)->centroid, query_point);
            if (distance < min_distance)
            {
                min_distance = distance;
                best_cluster = i;
            }
        }
        if (clusters->at(best_cluster)->most_common_class == query_point->get_label())
        {
            correct++;
        }
        // printf("Point %d assigned to cluster %d\n", query_point->get_label(), best_cluster);
    }
    return (correct / validation_data->size()) * 100;
}

int main()
{
    DataHandler *data_handler = new DataHandler();
    data_handler->read_feature_vector("../../data/train-images-idx3-ubyte");
    data_handler->read_feature_labels("../../data/train-labels-idx1-ubyte");
    data_handler->split_data();
    data_handler->count_classes();

    double performance = 0;
    double best_performance = 0;
    int best_k = 1;

    for (int k = data_handler->get_num_classes(); k < data_handler->get_training_data()->size() * .1; k++)
    {
        KMeans *k_means = new KMeans(k);
        k_means->set_training_data(data_handler->get_training_data());
        k_means->set_test_data(data_handler->get_test_data());
        k_means->set_validation_data(data_handler->get_validation_data());

        k_means->init_clusters();
        k_means->train();
        performance = k_means->validate();
        printf("K: %d Performance: %f\n", k, performance);

        if (performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }

        delete k_means;
    }

    KMeans *k_means = new KMeans(best_k);
    k_means->set_training_data(data_handler->get_training_data());
    k_means->set_test_data(data_handler->get_test_data());
    k_means->set_validation_data(data_handler->get_validation_data());

    k_means->init_clusters();
    performance = k_means->test();
    printf("K: %d, Test Performance: %f\n", best_k, performance);

    delete k_means;
}