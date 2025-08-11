#ifndef _H_CIRCUIT
#define _H_CIRCUIT

#include "string"
using std::string;

const string networkinputfile = "network";
const string baseinputfile = "./testdata/decompled/img_";
const int INPUT_NODES = 784;
const int one = 2147483647, zero = 2147483646;
const int TESTS = 100;

#pragma region declarations

inline int32_t __read_int32_t(std::ifstream &in);

#endif
