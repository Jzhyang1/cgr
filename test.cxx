#include <vector>
#include <cstdio>
#include <memory>
#include <chrono>
#include <map>
#include <math.h>
#include <cctype>
#include "tensors.h"
#include <iostream> // Required for input/output operations (e.g., cout)
#include <fstream>  // Required for file stream operations (e.g., ifstream)


using namespace std;

int main() {
    Matrix<float, 3, 3> m1;
    //  1  3  2
    //  2  1  1
    //  4  5  1
    m1[0, 0] = 1.0f;
    m1[0, 1] = 3.0f;
    m1[0, 2] = 2.0f;
    m1[1, 0] = 2.0f;
    m1[1, 1] = 1.0f;
    m1[1, 2] = 1.0f;
    m1[2, 0] = 4.0f;
    m1[2, 1] = 5.0f;
    m1[2, 2] = 1.0f;

    cout << m1 * m1 << endl;
}