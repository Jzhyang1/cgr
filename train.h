#include "model.h"
#include <map>

constexpr int alphsize = 32;
constexpr int lookback = 32;
constexpr int batchsize = 100;
constexpr int epochs = 1;

SequentialLayers<alphsize, lookback * alphsize, 
    ReLU<256>, Tanh<128>, Tanh<64>, Softmax<32>
> base_model(0.02f); // predicts the next letter


std::map<char, int> alphabet;
std::map<int, char> inverse_alphabet;

void init_alphabet() {
    //define our alphabet
    for (char c = 'a'; c <= 'z'; ++c) {
        alphabet[c] = alphabet.size();
    }
    alphabet[' '] = alphabet.size();
    alphabet['\xA0'] = alphabet.size();
    alphabet[','] = alphabet.size();
    alphabet[':'] = alphabet.size();
    alphabet['.'] = alphabet.size();
    alphabet['\n'] = alphabet.size();
    for (auto entry : alphabet) {
        inverse_alphabet[entry.second] = entry.first;
    }
}