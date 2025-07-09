#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_internals.h"

void test_init_input_layer() {
    int input_size = 4;
    Layer* layer = init_input(input_size);

    assert(layer != NULL);
    assert(layer->type == INPUT);
    assert(layer->input_size == input_size);
    assert(layer->output_size == input_size);
    assert(layer->input != NULL);
    assert(layer->output != NULL);

    free(layer->input);
    free(layer->output);
    free(layer);
    printf("test_init_input_layer passed\n");
}

void test_init_dense_layer() {
    int input_size = 3;
    int output_size = 2;
    Layer* layer = init_dense(output_size, input_size);

    assert(layer != NULL);
    assert(layer->type == DENSE);
    assert(layer->input_size == input_size);
    assert(layer->output_size == output_size);
    assert(layer->input != NULL);
    assert(layer->output != NULL);
    assert(layer->weights != NULL);
    assert(layer->biases != NULL);

    for (int i = 0; i < output_size; i++) {
        assert(layer->weights[i] != NULL);
    }

    // Clean up
    for (int i = 0; i < output_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
    free(layer->input);
    free(layer->output);
    free(layer);

    printf("test_init_dense_layer passed\n");
}

void test_create_and_destroy_net() {
    int num_layers = 3;
    int layer_sizes[3] = {4, 3, 2};
    double lr = 0.2;
    LayerType layer_types[3] = {INPUT, DENSE, DENSE};
    Activation activations[] = {NONE, RELU, RELU};
    NeuralNet* net = init_net(num_layers, layer_sizes, layer_types, activations, lr);
    assert(net != NULL);
    assert(net->num_layers == num_layers);
    assert(net->lr == lr);

    for (int i = 0; i < num_layers; i++) {
        assert(net->layers[i] != NULL);
    }

    destroy_net(net);
    printf("test_create_and_destroy_net passed\n");
}

int main() {
    nn_init();  

    test_init_input_layer();
    test_init_dense_layer();
    test_create_and_destroy_net();

    printf("All tests passed!\n");
    return 0;
}
