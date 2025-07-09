#include <stdio.h>
#include "nn.h" 

int main(void) {
    nn_init();
    
    double inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    double targets[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    int layer_sizes[] = {2, 3, 1};
    double lr = 0.1;
    LayerType layer_types[] = {INPUT, DENSE, DENSE};
    Activation activations[] = {NONE, SIGMOID, SIGMOID};

    NeuralNet* nn = init_net(3, layer_sizes, layer_types, activations, lr);
    
    for (int epoch = 0; epoch < 100000; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++) {
            network_forward(nn, inputs[i]);
            double loss = calculate_loss(nn, targets[i]);
            total_loss += loss;
            network_backward(nn, targets[i]);
        }

        if (epoch % 5000 == 0) {
            printf("Epoch %d, Loss: %.6f\n", epoch, total_loss / 4.0);
        }
    }

    print_network_weights(nn);

    printf("\n=== XOR Results ===\n");
    for (int i = 0; i < 4; i++) {
        network_forward(nn, inputs[i]);
        printf("Input: [%.1f, %.1f] -> Output: %.4f (Target: %.1f)\n",
               inputs[i][0], inputs[i][1], nn->layers[2]->output[0], targets[i][0]);
    }

    destroy_net(nn);

    return 0;
}
