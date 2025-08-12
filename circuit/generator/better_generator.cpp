#include <iostream>
#include <fstream>
#include <cassert>
#include <random>

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 28 * 28);

    int N = (1 << 10) * 10;

    std::cout << N * 2 - 10 << "\n";

    int minimum = 28 * 28;
    int maximum = 28 * 28;

    while (minimum + N > maximum) {
        std::cout << dis(gen) << " " << dis(gen) << "\n";
        ++maximum;
    }

    while (minimum + 10 < maximum) {
        std::cout << minimum++ << " " << minimum++ << "\n";
        maximum++;
    }
}