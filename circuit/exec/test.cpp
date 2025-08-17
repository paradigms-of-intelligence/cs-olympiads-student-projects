#include<fstream>
#include<vector>
#include<map>
#include<queue>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include "../circuit.h"

#ifndef INT32_MAX
typedef int int32_t;
#endif

//The sus ඞඞඞඞඞ

/* NOTES:

This code assumes that the first INPUT_NODES nodes 
are sorted and their id is in range [1, INPUT_NODES].

It also assumes that the last 10 nodes are sorted and 
are mapped to [0, 9].
*/

#pragma region declarations

//Beginning of declarations
struct Node {
    int32_t type;
    int32_t id;
    int32_t link_a;
    int32_t link_b;
    
    //Default constructor
    Node(int32_t _type, int32_t _id, int32_t _link_a, int32_t _link_b): 
        type(_type), id(_id), link_a(_link_a),  link_b(_link_b) {}
};


//End of declarations
#pragma endregion
//#######################################
//#######################################
#pragma region program_logic

//Beginning of code

//Store nodes in toposorted order
std::vector<Node> input_nodes;
std::vector<Node> toposorted_nodes;
std::vector<Node> final_nodes;

//Number of new nodes, avoid clashes between ids
int32_t next_free_node;
int32_t node_count;
int32_t first_output_id;

void toposort_nodes() {
    //Find reverse links
    //TODO: change this please :(
    std::map<int32_t, std::vector<int32_t>> reverse_link;
    std::map<int32_t, int32_t> link_count;


    std::queue<int32_t> process_queue;
    for(size_t i = 1; i <= INPUT_NODES; i++) process_queue.push(i);

    
    for(Node& node : input_nodes)
    {
        if(abs(node.link_a) != ALWAYS_TRUE && abs(node.link_a) != ALWAYS_FALSE)
        {
            reverse_link[abs(node.link_a)].emplace_back(node.id);
            link_count[node.id]++;
        }

        if(abs(node.link_b) != ALWAYS_TRUE && abs(node.link_b) != ALWAYS_FALSE) 
        {
            reverse_link[abs(node.link_b)].emplace_back(node.id);
            link_count[node.id]++;
        }

        if(link_count[node.id] == 0) process_queue.push(node.id);
    }
    
    size_t processed_count = 0;

    while(!process_queue.empty()) { 
        int32_t node_id = process_queue.front();
        process_queue.pop();

        processed_count++;

        if(node_id > (int32_t)INPUT_NODES) 
        {
            toposorted_nodes.push_back(input_nodes[node_id-INPUT_NODES-1]);
        }

        for(int32_t edge : reverse_link[node_id])
        {
            if(--link_count[edge] == 0) process_queue.push(edge);
        }
    }

    if (toposorted_nodes.size() != static_cast<size_t>(node_count - INPUT_NODES))
    program_abort(EXIT_CONVERSION_ERROR);
}       


void replace_gates() {
    // not including input nodes

    for(size_t i = 0; i < toposorted_nodes.size(); i++) {
        std::vector<Node> new_nodes;

        int32_t __input_a = toposorted_nodes[i].link_a;    
        int32_t __input_b = toposorted_nodes[i].link_b;
        int32_t __id = toposorted_nodes[i].id;

        switch (toposorted_nodes[i].type) {
            case 0: // ALWAYS 0
                new_nodes.push_back(Node(0, __id, ALWAYS_FALSE, ALWAYS_FALSE));
                break;

            case 1: // AND
                new_nodes.push_back(Node(0, __id, __input_a, __input_b));
                break;

            case 2: // A AND !B
                new_nodes.push_back(Node(0, __id, __input_a, -__input_b));
                break;

            case 3: // A
                new_nodes.push_back(Node(0, __id, __input_a, ALWAYS_TRUE));
                break;

            case 4: // B AND !A
                new_nodes.push_back(Node(0, __id, -__input_a, __input_b));
                break;

            case 5: // B
                new_nodes.push_back(Node(0, __id, ALWAYS_TRUE, __input_b));
                break;

            case 6: // XOR
            {
                int32_t clean_gate = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, __input_a, __input_b));

                int32_t neg_gate = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, -__input_a, -__input_b));

                new_nodes.push_back(Node(0, __id, -clean_gate, -neg_gate));
                break;
                
            }

            case 7: // OR
            {

                int32_t nand_gate = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, -__input_a, -__input_b));

                new_nodes.push_back(Node(0, __id, -nand_gate, -nand_gate));
                break;
            }

            case 8: // NOR
            {

                int32_t t_gate = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, -__input_a, -__input_b));

                new_nodes.push_back(Node(0, __id, t_gate, t_gate));
                break;
            }

            case 9: // XNOR
            {
                int32_t nand_1 = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, __input_a, __input_b));

                int32_t nand_2 = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, __input_a, -nand_1));

                int32_t nand_3 = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, __input_b, -nand_1));

                int32_t nand_4 = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, -nand_2, -nand_3));

                new_nodes.push_back(Node(0, __id, nand_4, nand_4));

                break;  
            }

            case 10: // !B
                new_nodes.push_back(Node(0, __id, -__input_b, ALWAYS_TRUE));
                break;

            case 11: // A OR !B
            {

                int32_t nand_gate = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, -__input_a, __input_b));

                new_nodes.push_back(Node(0, __id, -nand_gate, -nand_gate));
                break;
            }

            case 12: // !A
                new_nodes.push_back(Node(0, __id, -__input_a, ALWAYS_TRUE));
                break;

            case 13: // B OR !A
            {

                int32_t nand_gate = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, __input_a, -__input_b));

                new_nodes.push_back(Node(0, __id, -nand_gate, -nand_gate));
                break;
            }

            case 14: // NAND
            {

                int32_t and_node = next_free_node;
                new_nodes.push_back(Node(0, next_free_node++, __input_a, __input_b));

                new_nodes.push_back(Node(0, __id, -and_node, -and_node));
                break;
            }

            case 15: // ALWAYS 1
                new_nodes.push_back(Node(0, __id, ALWAYS_TRUE, ALWAYS_TRUE));
                break;

            default:
                program_abort(EXIT_INVALID_NODE_DATA);
                break;
        }

        
        if(new_nodes.empty() || new_nodes.back().id != toposorted_nodes[i].id)
            program_abort(EXIT_CONVERSION_ERROR);

        for(size_t t = 0; t < new_nodes.size(); t++)
            final_nodes.push_back(new_nodes[t]);
    }
}

int main(int argc, char const *argv[]) {

    if(argc != 3) program_abort(EXIT_WRONG_USAGE);

    std::ifstream t16_ifstream(argv[1], std::ios::binary | std::ios::in);
    std::ofstream t2_ofstream(argv[2], std::ios::binary | std::ios::out);

    if(!t16_ifstream.is_open() || !t2_ofstream.is_open()) 
        program_abort(EXIT_FILE_ERROR);

    node_count = read_int32_t(t16_ifstream);

    first_output_id = node_count - OUTPUT_NODES + 1; 
    next_free_node = node_count + 1;

    for(int32_t i = 0; i < node_count-INPUT_NODES; ++i)
    {       
        int32_t node_value[4];
        node_value[0] = read_int32_t(t16_ifstream);
        node_value[1] = read_int32_t(t16_ifstream);
        node_value[2] = read_int32_t(t16_ifstream);
        node_value[3] = read_int32_t(t16_ifstream);
        if(node_value[0] > 16 || node_value[0] < 0
         || node_value[1] <=(int32_t) INPUT_NODES){
            program_abort(EXIT_INVALID_NODE_DATA);
        }

        input_nodes.push_back(Node(node_value[0], node_value[1], node_value[2], node_value[3]));
    }   

    std::sort(input_nodes.begin(), input_nodes.end(), [&](const Node& a, const Node& b)
    {
        return a.id < b.id;
    });

    toposort_nodes();   


    std::cout << "Testing\n";
    std::vector<bool> values(node_count+1, 0);
    std::ifstream in("../data/testdata.txt", std::ios::in);
    int correct;
    float ans = 0;
    for(int i = 0; i < 10000; i++)
    {
        string input, res;
        getline(in, input);
        getline(in, res);

        correct = res[0]-'0';
        for(int j = 1; j <= INPUT_NODES; j++)
        {
            values[j] = (input[j-1] == '1');
        }

        for(int j = INPUT_NODES+1; i <= node_count; i++)
        {
            bool a = values[toposorted_nodes[j-1-INPUT_NODES].link_a];
            bool b = values[toposorted_nodes[j-1-INPUT_NODES].link_b];
            int type = toposorted_nodes[j-1-INPUT_NODES].type;

            bool res;
            switch (type)
            {
            case 0:
                res = false;
                break;
            case 1:
                res = a && b;
                break;
            case 2:
                res = a && !b;
                break;
            case 3:
                res = a;
                break;
            case 4:
                res = !a && b;
                break;
            case 5:
                res = b;
                break;
            case 6:
                res = a != b;
                break;
            case 7:
                res = a || b;
                break;
            case 8:
                res = !(a || b);
                break;
            case 9:
                res = a == b;
                break;
            case 10:
                res = !b;
                break;
            case 11:
                res = a || !b;
                break;
            case 12:
                res = !a;
                break;
            case 13:
                res = a || b;
                break;
            case 14:
                res = !(a && b);
                break;
            case 15:
                res = true;
                break;
            default:
                break;
            }

            values[toposorted_nodes[j-1-INPUT_NODES].id] = res;
        }

        int first_output_node = node_count-1000;

        int sz = 100;
        std::vector<float> cnt(10, 0);
        for(int number = 0; number < 10; number++)
        {
            for(int j = 0; j < sz; j++)
            {
                if(values[first_output_node+number*sz+j] == 1)
                    cnt[number] += 0.01;
            }
        }

        int l = 0;
        for(int k = 1; k < 10; k++) if(cnt[l] > cnt[i]) l = k;

        if(l == correct) ans++;
    }   

    std::cout << ans/10000.0 << "\n";

    t16_ifstream.close();
    return 0;
}

#pragma endregion
