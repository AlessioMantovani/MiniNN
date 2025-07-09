#ifndef NN_INTERNAL_H
#define NN_INTERNAL_H

#include "nn.h" 

Layer* init_dense(int layer_size, int previous_layer_size);
Layer* init_input(int input_size);
void setup_layer_functions(Layer* layer);

#endif
