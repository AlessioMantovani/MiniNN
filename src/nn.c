#include "nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <math.h>

void nn_init() {
    srand((unsigned int)time(NULL));
}

void randomize_w(double** w, int layer_size, int previous_layer_size) {
    for (int i = 0; i < layer_size; i++) {
        for (int j = 0; j < previous_layer_size; j++) {
            w[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
}

void randomize_b(double* b, int layer_size) {
    for (int i = 0; i < layer_size; i++) {
        b[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

Layer* init_dense(int layer_size, int previous_layer_size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) {
        printf("Error while creating layer\n Exiting...\n");
        exit(1);
    }
    layer->type = DENSE;
    layer->input_size = previous_layer_size;
    layer->output_size = layer_size;

    layer->input = (double*)calloc(previous_layer_size, sizeof(double));
    layer->output = (double*)calloc(layer_size, sizeof(double));

    // init weights
    layer->weights = (double**)malloc(layer_size * sizeof(double*));
    for (int i = 0; i < layer_size; i++) {
        layer->weights[i] = (double*)malloc(previous_layer_size * sizeof(double));
    }
    randomize_w(layer->weights, layer_size, previous_layer_size);
    // init bias
    layer->biases = (double*)malloc(layer_size * sizeof(double));
    randomize_b(layer->biases, layer_size);
    
    return layer;
}

Layer* init_input(int input_size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) {
        printf("Error while creating input layer\n Exiting...\n");
        exit(1);
    }

    layer->type = INPUT;
    layer->input_size = input_size;
    layer->output_size = input_size;

    layer->input = (double*)calloc(input_size, sizeof(double));
    layer->output = (double*)calloc(input_size, sizeof(double));

    layer->weights = NULL;
    layer->biases = NULL;

    layer->forward = NULL;
    layer->backward = NULL;
    layer->activate = NULL;
    layer->activate_derivative = NULL;

    return layer;
}

NeuralNet* create_net(int num_layers, int* layer_sizes, LayerType* layer_types, double lr) {
    NeuralNet* nn = (NeuralNet*)malloc(sizeof(NeuralNet));
    if (!nn) {
        printf("Error while creating Neural net\n Exiting...\n");
        exit(1);
    }

    nn->num_layers = num_layers;
    nn->lr = lr;
    
    nn->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    if (!nn->layers) {
        printf("Error allocating layers array\n Exiting...\n");
        exit(1);
    }
    
    for (int i = 0; i < num_layers; i++) {
        switch (layer_types[i]) {
        case INPUT:
            nn->layers[i] = init_input(layer_sizes[i]);
            break;            
        case DENSE:
            nn->layers[i] = init_dense(layer_sizes[i], layer_sizes[i - 1]);
            break;
        
        default:
            printf("Unknown layer type at index %d\n", i);
            exit(1);
        }
    }
    return nn;
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double identity(double x) {
    return x;
}

double identity_derivative(double x) {
    return 1.0;
}

double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

double leaky_relu(double x) {
    return x > 0 ? x : 0.01 * x;
}

double leaky_relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.01;
}

void input_forward(Layer* layer) {
    // Input layer just copies input to output
    for (int i = 0; i < layer->input_size; i++) {
        layer->output[i] = layer->input[i];
    }
}

void dense_forward(Layer* layer) {
    // Compute: output = activation(weights * input + bias)
    for (int i = 0; i < layer->output_size; i++) {
        double sum = 0.0;
        
        // Compute weighted sum
        for (int j = 0; j < layer->input_size; j++) {
            sum += layer->weights[i][j] * layer->input[j];
        }
        
        // Add bias
        sum += layer->biases[i];
        
        // Store pre-activation value for backward pass
        layer->pre_activation[i] = sum;
        
        // Apply activation function using function pointer
        if (layer->activate) {
            layer->output[i] = layer->activate(sum);
        } else {
            // Default to identity if no activation is set
            layer->output[i] = sum;
        }
    }
}

void input_backward(Layer* layer, double* output_gradient, double learning_rate) {
    // Input layer just passes gradients through
    for (int i = 0; i < layer->input_size; i++) {
        layer->input_gradient[i] = output_gradient[i];
    }
}

void dense_backward(Layer* layer, double* output_gradient, double learning_rate) {
    // Initialize input gradient to zero
    for (int i = 0; i < layer->input_size; i++) {
        layer->input_gradient[i] = 0.0;
    }
    
    // Compute gradients and update parameters
    for (int i = 0; i < layer->output_size; i++) {
        // Apply activation derivative using function pointer
        double activation_gradient = 1.0; // Default to 1 if no derivative function
        if (layer->activate_derivative) {
            activation_gradient = layer->activate_derivative(layer->pre_activation[i]);
        }
        
        double delta = output_gradient[i] * activation_gradient;
        
        // Update bias
        layer->biases[i] -= learning_rate * delta;
        
        // Update weights and compute input gradient
        for (int j = 0; j < layer->input_size; j++) {
            // Accumulate gradient w.r.t. input (for backpropagation to previous layer)
            layer->input_gradient[j] += layer->weights[i][j] * delta;
            
            // Update weight using gradient descent
            layer->weights[i][j] -= learning_rate * delta * layer->input[j];
        }
    }
}

void setup_activation_functions(Layer* layer, Activation activation) {
    switch (activation) {
        case RELU:
            layer->activate = relu;
            layer->activate_derivative = relu_derivative;
            break;
            
        case SIGMOID:
            layer->activate = sigmoid;
            layer->activate_derivative = sigmoid_derivative;
            break;

        case TANH:
            layer->activate = tanh;
            layer->activate_derivative = tanh_derivative;
            break;

        case LEAKY_RELU:
            layer->activate = leaky_relu;
            layer->activate_derivative = leaky_relu_derivative;
            break;

        case NONE:
        default:
            layer->activate = identity;
            layer->activate_derivative = identity_derivative;
            break;
    }
}

void setup_layer_functions(Layer* layer, Activation activation) {
    switch (layer->type) {
        case INPUT:
            layer->forward = input_forward;
            layer->backward = input_backward;
            layer->activate = identity;
            layer->activate_derivative = identity_derivative;
            break;
            
        case DENSE:
            layer->forward = dense_forward;
            layer->backward = dense_backward;
            setup_activation_functions(layer, activation);
            break;
            
        default:
            printf("Unknown layer type\n");
            exit(1);
    }
}

void network_forward(NeuralNet* nn, double* input) {
    // Set input to the first layer
    for (int i = 0; i < nn->layers[0]->input_size; i++) {
        nn->layers[0]->input[i] = input[i];
    }
    
    // Forward pass through all layers
    for (int i = 0; i < nn->num_layers; i++) {
        Layer* layer = nn->layers[i];
        
        // Call forward function
        if (layer->forward) {
            layer->forward(layer);
        }
        
        // Copy output to next layer's input (except for last layer)
        if (i < nn->num_layers - 1) {
            for (int j = 0; j < layer->output_size; j++) {
                nn->layers[i + 1]->input[j] = layer->output[j];
            }
        }
    }
}

void network_backward(NeuralNet* nn, double* target_output) {
    // Calculate output error (MSE derivative: 2 * (output - target))
    Layer* output_layer = nn->layers[nn->num_layers - 1];
    
    // Initialize output gradient for the last layer
    for (int i = 0; i < output_layer->output_size; i++) {
        output_layer->output_gradient[i] = 2.0 * (output_layer->output[i] - target_output[i]);
    }
    
    // Backward pass through all layers (from last to first, excluding input layer)
    for (int i = nn->num_layers - 1; i >= 1; i--) {
        Layer* layer = nn->layers[i];
        
        if (layer->backward) {
            // Pass the output gradient to the layer's backward function
            layer->backward(layer, layer->output_gradient, nn->lr);
        }
        
        // Propagate gradients to previous layer
        if (i > 0) {
            Layer* prev_layer = nn->layers[i - 1];
            // Copy input gradient to previous layer's output gradient
            for (int j = 0; j < prev_layer->output_size; j++) {
                prev_layer->output_gradient[j] = layer->input_gradient[j];
            }
        }
    }
}

double calculate_loss(NeuralNet* nn, double* target_output) {
    Layer* output_layer = nn->layers[nn->num_layers - 1];
    double loss = 0.0;
    
    for (int i = 0; i < output_layer->output_size; i++) {
        double diff = output_layer->output[i] - target_output[i];
        loss += diff * diff;
    }
    
    return loss / output_layer->output_size;
}

void print_network_weights(NeuralNet* nn) {
    printf("=== Network Weights ===\n");
    for (int i = 1; i < nn->num_layers; i++) {
        Layer* layer = nn->layers[i];
        if (layer->type == DENSE) {
            printf("Layer %d (Dense):\n", i);
            printf("  Weights:\n");
            for (int j = 0; j < layer->output_size; j++) {
                printf("    Neuron %d: [", j);
                for (int k = 0; k < layer->input_size; k++) {
                    printf("%.4f", layer->weights[j][k]);
                    if (k < layer->input_size - 1) printf(", ");
                }
                printf("]\n");
            }
            printf("  Biases: [");
            for (int j = 0; j < layer->output_size; j++) {
                printf("%.4f", layer->biases[j]);
                if (j < layer->output_size - 1) printf(", ");
            }
            printf("]\n");
        }
    }
    printf("======================\n");
}

NeuralNet* init_net(int num_layers, int* layer_sizes, LayerType* layer_types, Activation* activations, double lr) {
    NeuralNet* nn = create_net(num_layers, layer_sizes, layer_types, lr);
    // Set up function pointers and allocate gradient memory for each layer
    for (int i = 0; i < num_layers; i++) {
        Layer* layer = nn->layers[i];
        layer->activation = activations[i];
        // Allocate gradient memory
        layer->input_gradient = (double*)calloc(layer->input_size, sizeof(double));
        layer->output_gradient = (double*)calloc(layer->output_size, sizeof(double));
        
        if (layer->type == DENSE) {
            // Allocate memory for pre-activation values
            layer->pre_activation = (double*)calloc(layer->output_size, sizeof(double));
        } else {
            layer->pre_activation = NULL;
        }
        
        // Set up function pointers (including activation functions)
        setup_layer_functions(layer, activations[i]);
    }
    
    return nn;
}

void destroy_net(NeuralNet* nn) {
    if (!nn) return;
    
    for (int i = 0; i < nn->num_layers; i++) {
        Layer* layer = nn->layers[i];
        if (!layer) continue;
        
        // Free original memory
        free(layer->input);
        free(layer->output);
        
        // Free gradient memory
        free(layer->input_gradient);
        free(layer->output_gradient);
        free(layer->pre_activation);
        
        if (layer->type == DENSE) {
            for (int j = 0; j < layer->output_size; j++) {
                free(layer->weights[j]);
            }
            free(layer->weights);
            free(layer->biases);
        }
        free(layer);
    }
    free(nn->layers);
    free(nn);
}