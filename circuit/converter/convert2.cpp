#include <fstream>
#include <vector>
#include <iostream>

#ifdef NDEBUG
    #define assert(expression) 0

#else
    #define assert(expression) { if(!expression) { std::cerr << "what: " #expression << "\nfile " << (__FILE__) << ", line " << (__LINE__) << std::endl; std::abort(); }}

#endif

typedef int int32_t;

int get_int(std::ifstream &in) {
    int32_t eingabe;
    // in.read(reinterpret_cast<char*>(&eingabe), sizeof(int32_t));
    in >> eingabe;

    return eingabe;
}

struct Node {
    int left;
    int right;

    int id;

    Node(int left, int right, int id) : left(left), right(right), id(id) {};
    Node(int id) : Node(-1, -1, id) {};
    Node() : Node(-1) {};
};

int NUMBER_OF_INPUT_NODES = 2;

int main() {
    // std::ios::binary
    std::ifstream in("input.txt");
    assert(!in.fail());

    int32_t node_count_input = get_int(in);

    std::vector<Node> return_nodes(NUMBER_OF_INPUT_NODES);

    for (int i = 0; i < NUMBER_OF_INPUT_NODES; ++i) {
        return_nodes[i] = {(i + 1) * 2};
        node_mapper[i] = (i + 1) * 2;
    }

    for (int i = NUMBER_OF_INPUT_NODES; i < node_count_input; ++i) {
        int type = get_int(in);
        int id = get_int(in);
        int left = get_int(in);
        int right = get_int(in);

        left = (left < 0 ? -2 * left : 2 * left);
        right = (right < 0 ? -2 * right : 2 * right);

        return_nodes.emplace_back(return_nodes[left / 2 - 1].id, return_nodes[right / 2 - 1].id, (int)return_nodes.size() * 2 + 2);
        return_nodes.emplace_back(return_nodes[left / 2 - 1].id, return_nodes[right / 2 - 1].id ^ 1, (int)return_nodes.size() * 2 + 2);
        return_nodes.emplace_back(return_nodes[left / 2 - 1].id ^ 1, return_nodes[right / 2 - 1].id, (int)return_nodes.size() * 2 + 2);
        return_nodes.emplace_back(return_nodes[left / 2 - 1].id ^ 1, return_nodes[right / 2 - 1].id ^ 1, (int)return_nodes.size() * 2 + 2);

        return_nodes.emplace_back(return_nodes[((int)return_nodes.size() - 4)].id ^ (type && 1), return_nodes[((int)return_nodes.size() - 3)].id ^ ((type >> 1) && 1), (int)return_nodes.size() * 2 + 2);
        return_nodes.emplace_back(return_nodes[((int)return_nodes.size() - 3)].id ^ ((type >> 2) && 1), return_nodes[((int)return_nodes.size() - 2)].id ^ ((type >> 3) && 1), (int)return_nodes.size() * 2 + 2);
        return_nodes[id - 1] = {return_nodes[((int)return_nodes.size() - 2)].id ^ 1, return_nodes[((int)return_nodes.size() - 1)].id ^ 1, id * 2 + 1};
    }

    int num_output_nodes = get_int(in);

    std::ofstream out("out.txt");
    out << "aag " << return_nodes.size() << " " << NUMBER_OF_INPUT_NODES << " 0 " << num_output_nodes << " " << return_nodes.size() - NUMBER_OF_INPUT_NODES << "\n";

    for (int i = 0; i < NUMBER_OF_INPUT_NODES; ++i) {
        out << (i + 1) * 2 << "\n";
    }

    for (int i = NUMBER_OF_INPUT_NODES; i < return_nodes.size(); ++i) {
        out << return_nodes[i].id << " " << return_nodes[i].left << " " << return_nodes[i].right << "\n";
    }

    for (int i = 0; i < num_output_nodes; ++i) {
        out << return_nodes[get_int(in)].id << "\n";
    }
}