#include <bits/stdc++.h>
#include "../circuit.h"
using namespace std;

struct AndNot_network {
    int N;
    vector<bool> value; // bit values for calculation
    vector<int> C_1, C_2; // edges
    vector<int> result_nodes; // ordered-ids of the final network output nodes

    void init(const char* path) {
        // initialize current network
        ifstream graphinput(path, ios::binary | ios::in);
        N = read_int32_t(graphinput);
        value.resize(N+1); C_1.resize(N+1); C_2.resize(N+1);

        for (int i = 1; i <= (int)(N-INPUT_NODES); i++) {
            int id = read_int32_t(graphinput);
            C_1[id] = read_int32_t(graphinput);
            C_2[id] = read_int32_t(graphinput);
        }

        result_nodes.resize(10);
        for (int i = 0; i < 10; i++) result_nodes[i] = read_int32_t(graphinput);


        graphinput.close();
    }

    void input_into(vector<bool> &in) {
        for (int i = 1; i <= (int)INPUT_NODES; i++) value[i] = in[i-1];
    }


    bool getvalue(int id){
        bool c = 0;
        if (id < 0) {
            assert(-id <= N);
            c = !value[-id];
        }
        else if (id == ALWAYS_TRUE) c = 1;
        else if (id == ALWAYS_FALSE) c = 0;
        else {
            assert( id <= N );
            c = value[id];
        }
        return c;
    }

    void calculatenetwork() {
        // calculate all node values
        for (int i = INPUT_NODES+1; i <= N; i++) {
            value[i] = getvalue(C_1[i]) & getvalue(C_2[i]);
        }
    }

    int guess() {
        // return the network's guess for MNIST input
        int g = -1;

        for (int counter = 0; counter < 10; counter++) {
            if (value[result_nodes[counter]-INPUT_NODES]) {
                if(g != -1) return -1;
                g = counter;
            }
        }

        return g;
    }
};

bool make_test(int n, AndNot_network &net, const char* directory) {
    // 1 if test correct, 0 if not
    //to fix, variable `n` is useless
    string path = directory;
    path += "img_" + to_string(n) + ".txt";
    
    ifstream current_input(path, ios::in);
    vector<bool> input_values(INPUT_NODES);
    string input_image;

    getline(current_input, input_image);

    for (int i = 0; i < (int)INPUT_NODES; i++) {
        input_values[i] = (input_image[i] == '1');
    }

    net.input_into(input_values);

    net.calculatenetwork();

    //get the correct result and compare
    string correct;
    getline(current_input, correct);

    int gs = net.guess();
    return correct[0] == ('0'+gs);
}   

int main(int argc, char const *argv[]) {

    if(argc != 3) 
        program_abort(-1);

    // create the network structure
    AndNot_network net;
    net.init(argv[2]);

    // test on the test data
    float num = 0;
    for (int i = 0; i < TESTS; i++) {
        if (make_test(i, net, argv[1])) num++;
    }

    cout << num/(float)TESTS << endl;
}