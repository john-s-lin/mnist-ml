#ifndef K_MEANS_H
#define K_MEANS_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_set>
#include "common.hpp"
#include "data_handler.hpp"

typedef struct Cluster
{

    std::vector<double> *centroid;
    std::vector<Data *> *points;
    std::map<int, int> class_count;
    int most_common_class;

    Cluster(Data *initial_point)
    {
        centroid = new std::vector<double>();
        points = new std::vector<Data *>();
        for (auto val : *(initial_point->get_feature_vector()))
        {
            centroid->push_back(val);
        }
        points->push_back(initial_point);
        class_count[initial_point->get_label()] = 1;
        most_common_class = initial_point->get_label();
    }

    void add_to_cluster(Data *point)
    {
        int previous_size = points->size();
        points->push_back(point);
        for (unsigned i = 0; i < centroid->size(); i++)
        {
            centroid->at(i) = (centroid->at(i) * previous_size + point->get_feature_vector()->at(i)) / points->size();
        }
        if (class_count.find(point->get_label()) == class_count.end())
        {
            class_count[point->get_label()] = 1;
        }
        else
        {
            class_count[point->get_label()]++;
        }
        set_most_common_class();
    }

    void set_most_common_class()
    {
        int best_class;
        int most_freq = 0;
        for (auto it = class_count.begin(); it != class_count.end(); it++)
        {
            if (it->second > most_freq)
            {
                most_freq = it->second;
                best_class = it->first;
            }
        }
        most_common_class = best_class;
    }
} cluster_t;

class KMeans : public CommonData
{
    int k;
    std::vector<cluster_t *> *clusters;
    std::unordered_set<int> *used_points;

public:
    KMeans(int k);
    ~KMeans();

    void init_clusters();
    void init_clusters_for_class();
    void train();
    double euclidean_distance(std::vector<double> *centroid, Data *point);
    double validate();
    double test();
};

#endif // K_MEANS_H