#ifndef NN_H
#define NN_H

typedef enum {
    INPUT,
    DENSE
} LayerType;

// Forward declaration
typedef struct Layer Layer;

// Function pointer types
typedef void (*ForwardFunc)(Layer* layer);
typedef void (*BackwardFunc)(Layer* layer, double* output_gradient, double learning_rate);

typedef struct Layer {
    LayerType type;
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
} Layer;

typedef struct {
    int num_layers;
    Layer** layers;
} NeuralNet;

// Function declarations
void nn_init(void);
void randomize_w(double** w, int layer_size, int previous_layer_size);
void randomize_b(double* b, int layer_size);
Layer* init_dense(int layer_size, int previous_layer_size);
Layer* init_input(int input_size);
NeuralNet* init_net(int num_layers, int* layer_sizes, LayerType* layer_types);
void destroy_net(NeuralNet* nn);

// Layer functions
void input_forward(Layer* layer);
void input_backward(Layer* layer, double* output_gradient, double learning_rate);
void dense_forward(Layer* layer);
void dense_backward(Layer* layer, double* output_gradient, double learning_rate);
void setup_layer_functions(Layer* layer);

// Network functions
void network_forward(NeuralNet* nn, double* input);
void network_backward(NeuralNet* nn, double* target_output, double learning_rate);
double calculate_loss(NeuralNet* nn, double* target_output);
void print_network_weights(NeuralNet* nn);

// Activation functions
double relu(double x);
double relu_derivative(double x);

#endif