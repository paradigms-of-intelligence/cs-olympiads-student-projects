#ifndef _H_CIRCUIT
#define _H_CIRCUIT

#include "string"
#include <cstdint>

using std::string;

const char* NETWORK2_FILE_NAME = "network.bin";
const char* NETWORK16_FILE_NAME = "16_type_network.bin";
const char* BASE_INPUT_FILE = "./testdata/decompled/img_";
const int INPUT_NODES = 784;
const int32_t ALWAYS_TRUE = 2147483647;
const int32_t ALWAYS_FALSE = 2147483646;

const int TESTS = 100;

enum
{
    EXIT_WRONG_USAGE = 1,
    EXIT_FILE_ERROR = 2,
    EXIT_INVALID_NODE_DATA = 3,
    EXIT_TOPOSORT_FAILED = 4,
    EXIT_UNCONNECTED_NETWORK = 5,
    EXIT_CONVERSION_ERROR = 6
};

#pragma region declarations

inline int32_t read_int32_t(std::ifstream &in);
inline void write_int32_t(std::ofstream &out, int32_t data);

[[noreturn]] void program_abort(size_t exit_code);

#endif
