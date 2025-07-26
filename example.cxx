#include <vector>
#include <cstdio>
#include <iostream>
#include <memory>
#include <chrono>
#include <map>
#include <math.h>
#include "model.h"

using namespace std;

SequentialLayers<1, 2, 
    ReLU<2>, Tanh<4>, Tanh<4>, Tanh<1>
> easy_model(0.05f); // predicts if a number is greater than 50

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < 50000; ++j) {
        int i = rand() % 100;  // small range for learning
        float x = (float)i / 100.0f;  // normalized
        bool is_lesser = i < 50;

        ColVector<float, 2> input({x, 1});
        ColVector<float, 1> pred = easy_model.fwd(input);

        float error = pred[0] - is_lesser;
        ColVector<float, 1> dL_dy(DataSlice<float>({2.0f * error}));  // gradient of MSE

        easy_model.bwd(dL_dy);

        bool predicted_lesser = pred[0] > 0.5;
        cout << "prediction for " << i << " " << (predicted_lesser ? "<" : ">") << "50 @" << pred[0] << endl;
        cout << "loss: " << error * error << "\terror: " << error << endl;
    }

    int correct = 0;
    int total = 1000;
    for (int j = 0; j < total; ++j) {
        int i = rand() % 100;  // small range for learning
        float x = (float)i / 100.0f;  // normalized
        bool is_lesser = i < 50;

        ColVector<float, 2> input({x, 1});
        ColVector<float, 1> pred = easy_model.fwd(input);

        correct += (pred[0] > 0.5) == is_lesser;
    }
    cout << "Accuracy: " << correct << " of " << total << endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

}