#ifndef NN_H
#define NN_H

typedef enum {
    INPUT,
    DENSE
} LayerType;

typedef enum {
    NONE,
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU,
} Activation;

// Forward declaration
typedef struct Layer Layer;

// Function pointer types
typedef void (*ForwardFunc)(Layer* layer);
typedef void (*BackwardFunc)(Layer* layer, double* output_gradient, double learning_rate);
typedef double (*ActivationFunc)(double);
typedef double (*ActivationDerivativeFunc)(double);

typedef struct Layer {
    LayerType type;
    Activation activation;
    int input_size;
    int output_size;
    double* input;
    double* output;
    double** weights;
    double* biases;
    
    // Gradient storage
    
    // Gradient w.r.t. input
    double* input_gradient;   
    // Gradient w.r.t. output
    double* output_gradient;  
    // Store pre-activation values for backward pass
    double* pre_activation;   
    
    // Function pointers
    ForwardFunc forward;
    BackwardFunc backward;
    ActivationFunc activate;
    ActivationDerivativeFunc activate_derivative;
} Layer;

typedef struct {
    int num_layers;
    double lr;
    Layer** layers;
} NeuralNet;

// Function declarations
void nn_init(void);
NeuralNet* init_net(int num_layers, int* layer_sizes, LayerType* layer_types, Activation* activations, double lr);
void destroy_net(NeuralNet* nn);
void network_forward(NeuralNet* nn, double* input);
void network_backward(NeuralNet* nn, double* target_output);
double calculate_loss(NeuralNet* nn, double* target_output);
void print_network_weights(NeuralNet* nn);


#endif