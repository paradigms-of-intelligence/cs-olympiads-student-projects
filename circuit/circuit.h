#ifndef _H_CIRCUIT
#define _H_CIRCUIT

#include "string"
using std::string;

#ifndef INT32_MAX
typedef int int32_t;
#endif

const string networkinputfile = "network";
const string baseinputfile = "./testdata/decompled/img_";
const int INPUT_NODES = 784;
const int one = 2147483647, zero = 2147483646;
const int TESTS = 100;

#pragma region declarations

//Beginning of declarations
enum
{
    EXIT_WRONG_USAGE = 1,
    EXIT_FILE_ERROR = 2,
    EXIT_INVALID_NODE_DATA = 3,
    EXIT_TOPOSORT_FAILED = 4
};

inline int32_t __read_int32_t(std::ifstream &in);
void __program_abort(size_t exit_code);

#endif
