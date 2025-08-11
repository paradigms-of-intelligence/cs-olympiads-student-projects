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


void __program_abort(size_t exit_code)
{
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
    int32_t result = 0;
    try
    {
        in.read(reinterpret_cast<char*>(&result), 4);
        if(in.fail()) throw std::runtime_error("Reading failed");
    }
    catch(const std::exception& e)
    {
        std::printf(e.what());
        __program_abort(EXIT_FILE_ERROR);
    }
    return result;
}

#endif
