#include <bits/stdc++.h>
#include "circuit.h"
using namespace std;


void __program_abort(size_t exit_code) {
    std::printf("Error code: %d", exit_code);
    switch (exit_code)
    {
    case EXIT_WRONG_USAGE:
        std::printf("Usage: convert_network <path_to_type16>");
        break;
    case EXIT_FILE_ERROR:
        std::printf("Error in I/O processing.");
        break;
    case EXIT_INVALID_NODE_DATA:
        std::printf("Invalid node ID.");
        break;
    case EXIT_TOPOSORT_FAILED:
        std::printf("Toposort failed.");
        break;
    default:
        std::printf("Error. No additional information supplied.");
        break;
    }

    exit(1);
}

inline int32_t __read_int32_t(std::ifstream &in)
{
    int32_t result;
    in.read(reinterpret_cast<char*>(&result), 4);

    if (in.fail()) {
        std::printf("Reading failed");
        __program_abort(EXIT_FILE_ERROR);
    }

    return result;
}