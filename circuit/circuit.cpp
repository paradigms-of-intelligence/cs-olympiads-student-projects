#include <bits/stdc++.h>
#include "circuit.h"
using namespace std;

inline int32_t read_int32_t(std::ifstream &in)
{
    int32_t result = 0;
    in.read(reinterpret_cast<char*>(&result), 4);

    if(in.fail()) program_abort(EXIT_FILE_ERROR);
    return result;
}

inline void write_int32_t(std::ofstream &out, int32_t data) {
    out.write(reinterpret_cast<char*>(&data), 4);
    if(out.fail()) program_abort(EXIT_FILE_ERROR);
}

void program_abort(size_t exit_code) {
    std::fprintf(stderr, "Error code: %d\n", exit_code);
    switch (exit_code) {
    case EXIT_WRONG_USAGE:
        std::fprintf(stderr, "Usage: convert_network <t16_in_path> <t2_out_path>\n");
        break;
    case EXIT_FILE_ERROR:
        std::fprintf(stderr, "Error in I/O processing.\n");
        break;
    case EXIT_INVALID_NODE_DATA:
        std::fprintf(stderr, "Invalid node ID.\n");
        break;
    case EXIT_TOPOSORT_FAILED:
        std::fprintf(stderr, "Toposort failed.\n");
        break;
    case EXIT_UNCONNECTED_NETWORK:
        std::fprintf(stderr, "Network is unconnected: some node weren't visited during toposort routine.\n");
        break;
    case EXIT_CONVERSION_ERROR:
        std::fprintf(stderr, "An error was encountered during the conversion phase.\n");
        break;
    default:
        std::fprintf(stderr, "Error. No additional information supplied.\n");
        break;
    }

    exit(1);
}