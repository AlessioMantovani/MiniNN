# Neural Network Library in C

A lightweight, simple neural network library written in C for educational purposes and small projects. The library supports basic feedforward networks with customizable architectures and backpropagation training.

## Features

- ✅ Feedforward neural networks
- ✅ Dense (fully connected) layers
- ✅ Sigmoid activation function
- ✅ ReLU activation function
- ✅ Backpropagation training
- ✅ Mean Squared Error loss
- ✅ Memory management included

## Project Structure

```
neural-net-lib/
├── CMakeLists.txt
├── README.md
├── include/
│   └── nn.h              # Public API header
├── src/
│   └── nn.c              # Library implementation
├── examples/
│   └── xor_example.c     # XOR problem example
├── cmake/
│   ├── NeuralNetConfig.cmake.in
│   └── neuralnet.pc.in
└── tests/
    └── test_nn.c         # Unit tests (optional)
```

## Quick Start

### XOR Problem Example

```c
#include <stdio.h>
#include "nn.h"

int main(void) {
    // Initialize the library
    nn_init();
    
    // Define XOR training data
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double targets[4][1] = {{0}, {1}, {1}, {0}};
    
    // Create network: 2 inputs -> 3 hidden -> 1 output
    int layer_sizes[] = {2, 3, 1};
    double lr = 0.1;
    LayerType layer_types[] = {INPUT, DENSE, DENSE};
    Activation activations[] = {NONE, SIGMOID, SIGMOID};

    NeuralNet* nn = init_net(3, layer_sizes, layer_types, activations, lr);
    
    // Train the network
    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_loss = 0.0;
        
        for (int i = 0; i < 4; i++) {
            network_forward(nn, inputs[i]);
            double loss = calculate_loss(nn, targets[i]);
            total_loss += loss;
            network_backward(nn, targets[i], 0.5);
        }
        
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %.6f\n", epoch, total_loss / 4.0);
        }
    }
    
    // Test the trained network
    printf("\n=== Results ===\n");
    for (int i = 0; i < 4; i++) {
        network_forward(nn, inputs[i]);
        printf("Input: [%.0f, %.0f] -> Output: %.4f (Target: %.0f)\n",
               inputs[i][0], inputs[i][1], 
               nn->layers[2]->output[0], targets[i][0]);
    }
    
    // Clean up
    destroy_net(nn);
    return 0;
}
```

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-net-lib.git
cd neural-net-lib

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Install system-wide (optional)
make install
```

### Option 2: Manual Compilation

```bash
# Simple compilation for testing
gcc -o example examples/xor_example.c src/nn.c -Iinclude -lm

# Run the example
./example
```

## Usage

### Method 1: Using CMake (Recommended)

After installation, use the library in your CMake project:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find the library
find_package(NeuralNet REQUIRED)

# Create your executable
add_executable(my_program my_program.c)

# Link against the library
target_link_libraries(my_program NeuralNet::nn_static)
```

### Method 3: Manual Linking

```bash
# If installed system-wide
gcc -o my_program my_program.c -lnn -lm

# If using local build
gcc -o my_program my_program.c -I/path/to/include -L/path/to/lib -lnn -lm
```

## API Reference

### Core Functions

#### Network Management
```c
void nn_init(void);
NeuralNet* init_net(int num_layers, int* layer_sizes, LayerType* layer_types);
void destroy_net(NeuralNet* nn);
```

#### Training and Inference
```c
void network_forward(NeuralNet* nn, double* input);
void network_backward(NeuralNet* nn, double* target_output, double learning_rate);
double calculate_loss(NeuralNet* nn, double* target_output);
```

#### Utility Functions
```c
void print_network_weights(NeuralNet* nn);
```

### Data Types

#### LayerType
```c
typedef enum {
    INPUT,    
    DENSE     
} LayerType;
```

#### NeuralNet Structure
```c
typedef struct {
    int num_layers;
    Layer** layers;
} NeuralNet;
```

### Example Usage Patterns

#### Creating Different Network Architectures

```c
// Simple 2-layer network (input -> output)
int sizes1[] = {4, 1};
LayerType types1[] = {INPUT, DENSE};
NeuralNet* simple_net = init_net(2, sizes1, types1);

// Deep network with multiple hidden layers
int sizes2[] = {10, 20, 15, 5, 1};
LayerType types2[] = {INPUT, DENSE, DENSE, DENSE, DENSE};
NeuralNet* deep_net = init_net(5, sizes2, types2);
```

#### Training Loop Pattern

```c
for (int epoch = 0; epoch < epochs; epoch++) {
    double total_loss = 0.0;
    
    for (int i = 0; i < num_samples; i++) {
        // Forward pass
        network_forward(nn, training_inputs[i]);
        
        // Calculate loss
        double loss = calculate_loss(nn, training_targets[i]);
        total_loss += loss;
        
        // Backward pass
        network_backward(nn, training_targets[i], learning_rate);
    }
    
    // Optional: print progress
    if (epoch % 100 == 0) {
        printf("Epoch %d: Average Loss = %.6f\n", epoch, total_loss / num_samples);
    }
}
```

## Examples Output

When you run the XOR example, you should see output like:

```
Epoch 0, Loss: 0.250000
Epoch 1000, Loss: 0.124567
Epoch 2000, Loss: 0.045123
Epoch 3000, Loss: 0.012345
Epoch 4000, Loss: 0.003456
Epoch 5000, Loss: 0.001234
Epoch 6000, Loss: 0.000567
Epoch 7000, Loss: 0.000234
Epoch 8000, Loss: 0.000123
Epoch 9000, Loss: 0.000067

=== XOR Results ===
Input: [0, 0] -> Output: 0.0123 (Target: 0)
Input: [0, 1] -> Output: 0.9876 (Target: 1)
Input: [1, 0] -> Output: 0.9845 (Target: 1)
Input: [1, 1] -> Output: 0.0234 (Target: 0)
```

## Development

### Building for Development

```bash
# Quick compilation during development
gcc -o test examples/xor_example.c src/nn.c -Iinclude -lm -g -Wall -Wextra

# Run with debugging
gdb ./test
```

### Development Makefile

Create a `Makefile` for easier development:

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -g -O0 -Iinclude
LDFLAGS = -lm

xor_example: examples/xor_example.c src/nn.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: xor_example
	./xor_example

clean:
	rm -f xor_example

.PHONY: test clean
```

### Adding New Examples

1. Create a new `.c` file in the `examples/` directory
2. Include the header: `#include "nn.h"`
3. Compile: `gcc -o my_example examples/my_example.c src/nn.c -Iinclude -lm`

## Build Options

The CMake build system supports several options:

```bash
# Build with examples (default: ON)
cmake -DBUILD_EXAMPLES=ON ..

# Build with tests (default: ON)
cmake -DBUILD_TESTS=ON ..

# Release build
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build (default)
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Requirements

- **Compiler**: GCC 4.9+ or Clang 3.5+ (C99 support required)
- **Math Library**: libm (usually included with system)
- **CMake**: 3.10+ (for building)
- **OS**: Linux, macOS, Windows (with MinGW/MSYS2)

## Current Limitations

- Only supports sigmoid activation (ReLU code present but not used)
- Only Mean Squared Error loss function
- No GPU acceleration
- No regularization techniques
- No advanced optimizers (only basic gradient descent)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by educational neural network implementations
- Built for learning
- Focuses on simplicity and clarity over performance