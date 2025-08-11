#include <bits/stdc++.h>
#include "circuit.h"
using namespace std;

inline int32_t read_int32_t(std::ifstream &in)
{
    int32_t result;
    in.read(reinterpret_cast<char*>(&result), 4);

    assert(!in.fail()); // Loading File failed

    return result;
}