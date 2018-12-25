#include <fstream>
#include "object_ranker.h"
#include "json/json.hpp"

#include "ranker/dense.h"
#include "ranker/nalu.h"

using json = nlohmann::json;

void InitRanker(ranker_context& ranker, std::string filename) {
    std::ifstream i(filename);
    json j;
    i >> j;

    // for previous versions capabilty
    if (j.count("parameters")) {
        json additional_parameters = j["parameters"];
        ranker.use_abs = additional_parameters["use_abs"].get<bool>();
        ranker.num_layers = j.size()-1;
        assert(ranker.num_layers >= 1);
    } else {
        ranker.use_abs = false;
        ranker.num_layers = j.size();
    }
    ranker.layers = new Layer*[ranker.num_layers];
    std::string layer_prefix = "layer";
    int max_dim = 0;
    for(int i = 0; i < ranker.num_layers; i++) {
        auto curr_layer_name = layer_prefix + std::to_string(i);
        json curr_layer = j[curr_layer_name];
        if(i == 0) {
            ranker.input_dim = curr_layer["in_dim"];
            max_dim = curr_layer["in_dim"];
        }

        std::string layer_type;
        // for previous versions capabilty
        if (curr_layer.count("type")) {
            layer_type = curr_layer["type"];
        } else {
            layer_type = "DENSE";
        }

        if (layer_type == "DENSE") {
            std::vector<std::vector<float>> W = curr_layer["W"];
            std::vector<std::vector<float>> b = curr_layer["b"];
            int in_dim = curr_layer["in_dim"];
            int out_dim = curr_layer["out_dim"];
            float* biases = new float[out_dim];
            float* weights = new float[out_dim * in_dim];

            for(int j = 0; j < out_dim; j++) {
                biases[j] = b.at(j);
                for(int k = 0; k < in_dim; k++) {
                    weights[j * in_dim + k] = W.at(k).at(j);
                }
            }
            ranker.layers[i] = new Dense(in_dim,out_dim,weights,biases,curr_layer["activation"].get<std::string>());
        } else if (layer_type == "NALU") {
            std::vector<std::vector<float>> W = curr_layer["W"];
            std::vector<std::vector<float>> G = curr_layer["G"];
            int in_dim = curr_layer["in_dim"];
            int out_dim = curr_layer["out_dim"];
            float* gate = new float[out_dim * in_dim];
            float* weights = new float[out_dim * in_dim];

            for(int j = 0; j < out_dim; j++) {
                for(int k = 0; k < in_dim; k++) {
                    weights[j * in_dim + k] = W.at(k).at(j);
                    gate[j * in_dim + k] = G.at(k).at(j);
                }
            }
            ranker.layers[i] = new Nalu(in_dim,out_dim,weights,gate);
        }
        if(ranker.layers[i]->getOutputDim() > max_dim) {
            max_dim = ranker.layers[i]->getOutputDim();
        }
    }
    ranker.is_init = true;
    ranker.max_dim = max_dim;
}


void FreeRanker(ranker_context& ranker) {
    for(int i = 0; i < ranker.num_layers; i++) {
        ranker.layers[i]->free();
    }
    delete [] ranker.layers;
}


float EstimateRank(ranker_context& ranker, std::vector<double>& features) {
    if(!ranker.is_init) {
        return 0;
    }
    float *curr_buff = new float[ranker.max_dim];
    float *prev_buff = new float[ranker.max_dim];
    assert(features.size() == ranker.input_dim);
    for(int i = 0; i < ranker.input_dim; i++) {
        prev_buff[i] = ranker.use_abs ? std::abs(features[i]) : features[i];
    }
    //memcpy(prev_buff, prev_vector.data(), prev_vector.size() * sizeof(float));
    float result = 0;
    for(int i = 0; i < ranker.num_layers; i++) {
        ranker.layers[i]->process(prev_buff,curr_buff);
        float *tmp = curr_buff;
        curr_buff = prev_buff;
        prev_buff = tmp;
    }
    result = prev_buff[0];
    delete[] curr_buff;
    delete[] prev_buff;
    return result;
}
