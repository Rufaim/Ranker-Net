#ifndef OBJECT_RANKER
#define OBJECT_RANKER
#include <string>
#include <vector>

#include "ranker/layer.h"

struct ranker_context {
    Layer** layers;
    int num_layers;
    int input_dim;
    int max_dim;
    bool use_abs;
    bool is_init;
};

void InitRanker(ranker_context& ranker, std::string filename);

void FreeRanker(ranker_context& ranker);

float EstimateRank(ranker_context& ranker, std::vector<double>& features);

#endif // OBJECT_RANKER

