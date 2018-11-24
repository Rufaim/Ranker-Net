#include "ranker_utils.h"

#include <math.h>

void Activation(std::string func_name, float* val, int dim) {
    if (func_name == "R") { //ReLU
        for(int i = 0; i < dim; i++) {
            val[i] = val[i] > 0 ? val[i] : 0;
        }
    } else if (func_name == "S") { //Sigmoid
        for(int i = 0; i < dim; i++) {
            val[i] = 0.5*tanh(val[i])+1;
        }
    }
}
