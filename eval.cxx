#include "train.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

string seed = "For in that endless sleep what may come must give us pause, and ";

int main() {
    std::ifstream weights_file("weights.txt");

    init_alphabet();
    base_model.load_weights(weights_file);
    // base_model.save_weights(cout);

    for (int i = 0; i < 100; ++i) {
        vector<float> one_hot(lookback * alphsize);
        for (int i = seed.size() - lookback, j = 0; i < seed.size(); ++i, ++j) {
            one_hot[j*alphsize + alphabet[seed[i]]] = 1.0f;
        }
        one_hot[0] = 1.0f;

        ColVector<float, lookback * alphsize> input{DataSlice<float>(one_hot)};
        ColVector<float, alphsize> output = base_model.fwd(input);
        float p = 0;
        char c = '#';
        for (int i = 0; i < alphsize; ++i) {
            if (output[i] > p) {
                p = output[i];
                c = inverse_alphabet[i];
                cout << i << c << ',' << p << endl;
            }
        }
        seed += c;
        cout << seed << (int)c << endl;
    }
}