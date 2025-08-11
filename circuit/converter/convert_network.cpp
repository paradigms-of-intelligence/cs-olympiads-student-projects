#include<fstream>
#include<vector>
#include<map>

const size_t INPUT_SIZE = 784;
const int ALWAYS_TRUE = 2147483647;
const int ALWAYS_FALSE = 2147483646;

#pragma region declarations

//Beginning of declarations

enum {
    EXIT_WRONG_USAGE = 1,
    EXIT_FILE_ERROR = 2,
    EXIT_INVALID_NODE_DATA = 3,
    EXIT_TOPOSORT_FAILED = 4
};

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

inline int __read_int32_t(std::ifstream &in)
{
    int result = 0;

    in.read(reinterpret_cast<char*>(&result), 4);

    if (in.fail()) {
        std::printf("Reading failed");
        __program_abort(EXIT_FILE_ERROR);
    }
    return result;
}

struct Node
{
    int type;
    int id;
    int link_a;
    int link_b;
    
    //Default constructor
    Node(int _type, int _id, int _link_a, int _link_b) : type(_type), id(_id), link_a(_link_a), link_b(_link_b) {};
};


//End of declarations
#pragma endregion
//#######################################
//#######################################
#pragma region program_logic

//Beginning of code

//Store nodes in toposorted order
std::vector<Node> toposorted_nodes;
std::vector<Node> input_nodes;

//Number of new nodes, avoid clashes between ids
size_t offset = 0;
int node_count;

bool toposort_nodes()
{
    //TODO: change this please :(
    std::map<int, std::vector<int>> reverse_link;
    for(Node& node : input_nodes) {
        if(abs(node.link_a) == ALWAYS_FALSE || abs(node.link_b) == ALWAYS_TRUE) 
            continue;

        reverse_link[node.link_a].emplace_back(node.id);
        reverse_link[node.link_b].emplace_back(node.id);
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 2) __program_abort(EXIT_WRONG_USAGE);

    std::ifstream t16_ifstream;
    std::ofstream t2_ofstream;

    const char* t16_in_path = argv[1];
    const char* t2_out_path = argv[2];
    
    t16_ifstream = std::ifstream(t16_in_path, std::ios::binary | std::ios::in);
    t2_ofstream = std::ofstream(t2_out_path, std::ios::binary | std::ios::out);

    if (t16_ifstream.fail() || t2_ofstream.fail()) {
        __program_abort(EXIT_FILE_ERROR);
    }
    
    node_count = __read_int32_t(t16_ifstream);

    for(size_t i = 0; i < node_count; ++i)
    {       
        int node_value[4];
        node_value[0] = __read_int32_t(t16_ifstream);
        node_value[1] = __read_int32_t(t16_ifstream);
        node_value[2] = __read_int32_t(t16_ifstream);
        node_value[3] = __read_int32_t(t16_ifstream);

        if(node_value[0] > 16 || node_value[0] < 0
         || node_value[1] <= INPUT_SIZE 
         || node_value[2] <= INPUT_SIZE
         || node_value[3] <= INPUT_SIZE ){
            __program_abort(EXIT_INVALID_NODE_DATA);
        }

        input_nodes.emplace_back(node_value[0], node_value[1], node_value[2], node_value[3]);
    }   

    if(!toposort_nodes()) __program_abort(EXIT_TOPOSORT_FAILED);
    return 0;
}

#pragma endregion
