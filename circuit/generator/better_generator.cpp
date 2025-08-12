#include <iostream>
#include <fstream>
#include <cassert>
#include <random>

int main() {
    // Including the Random Stuff
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 28 * 28);

    // Defining a Size thingy
    int N = (1 << 10) * 10;

    // Calculating and giving out the size of the Network
    std::cout << N * 2 - 10 << "\n";

    // The First and Last Node of the Imaginary unused Nodes arrey.
    int minimum = 28 * 28;
    int maximum = 28 * 28;

    // Generating Nodes N times.
    while (minimum + N > maximum) {
        std::cout << dis(gen) << " " << dis(gen) << "\n";
        ++maximum;
    }

    // Merging all the Nodes so that we just have 10 Left. The outputs.
    while (minimum + 10 < maximum) {
        std::cout << minimum++ << " " << minimum++ << "\n";
        maximum++;
    }
}