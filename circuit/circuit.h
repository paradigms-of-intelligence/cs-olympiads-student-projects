#ifndef CIRCUIT_H
#define CIRCUIT_H
#pragma once

#include "string"
#include <cstdint>

using std::string;

extern const char* NETWORK2_FILE_NAME;
extern const char* NETWORK16_FILE_NAME;
extern const char* BASE_INPUT_FILE;
constexpr size_t INPUT_NODES = 784;
constexpr int32_t ALWAYS_TRUE = 2147483647;
constexpr int32_t ALWAYS_FALSE = 2147483646;

constexpr int TESTS = 100;

enum {
    EXIT_WRONG_USAGE = 1,
    EXIT_FILE_ERROR = 2,
    EXIT_INVALID_NODE_DATA = 3,
    EXIT_TOPOSORT_FAILED = 4,
    EXIT_UNCONNECTED_NETWORK = 5,
    EXIT_CONVERSION_ERROR = 6
};

int32_t read_int32_t(std::ifstream &in);
void write_int32_t(std::ofstream &out, int32_t data);

[[noreturn]] void program_abort(int32_t exit_code);

#endif
