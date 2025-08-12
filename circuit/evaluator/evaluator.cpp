#include <bits/stdc++.h>
#include "../circuit.h"
using namespace std;

struct AndNot_network {
    int N;
    vector<bool> value; //bit values for calculation
    vector<int> C_1, C_2; //edges
    vector<int> result_nodes; // ordered-ids of the final network output nodes

    void init() {
        // initialize current network
        ifstream graphinput(NETWORK2_FILE_NAME, ios::binary);
        N = read_int32_t(graphinput)+1;
        value.resize(N); C_1.resize(N); C_2.resize(N);

        for (int i = 0; i < N; i++) {
            int id = read_int32_t(graphinput);
            C_1[id] = read_int32_t(graphinput);
            C_2[id] = read_int32_t(graphinput);
        }

        result_nodes.resize(10);
        for (int i = 0; i < 10; i++) result_nodes[i] = read_int32_t(graphinput);
        graphinput.close();
    }

    void input_into(vector<bool> &in) {
        for (int i = 1; i <= INPUT_NODES; i++) value[i] = in[i-1];
    }

    int calculatenetwork() {
        // calculate all node values
        for (int i = INPUT_NODES+1; i < N; i++) {
            auto getvalue = [&](int id) -> bool{
                bool c;
                if (id < 0) {
                    assert(-id < N);
                    c = !value[-id];
                }
                else if (id == ALWAYS_TRUE) c = 1;
                else if (id == ALWAYS_FALSE) c = 0;
                else c = value[id];
                return c;
            };
            value[i] = getvalue(C_1[i]) & getvalue(C_2[i]);
        }
    }

    int guess() {
        // return the network's guess for MNIST input
        int g = -1;

        for (int counter = 0; counter < 10; counter++) {
            if (value[result_nodes[counter]]) {
                assert(g == -1); //the network returned more than 1 guess
                g = counter;
            }
        }

        assert(g != -1); // the network didn't guess
        return g;
    }
};

bool make_test(int n, AndNot_network &net) {
    // 1 if test correct, 0 if not
    //to fix, variable `n` is useless
    ifstream current_input(BASE_INPUT_FILE);
    vector<bool> input_values;
    string input_image;
    getline(current_input, input_image);

    for (int i = 0; i < INPUT_NODES; i++) {
        input_values[i] = (input_image[i] == '1');
    }

    net.input_into(input_values);
    net.calculatenetwork();

    //get the correct result and compare
    string correct;
    getline(current_input, correct);
    return correct[0] == ('0'+net.guess());
}   


int main() {
    // create the network structure
    AndNot_network net;
    net.init();

    // test on the test data
    float num = 0;
    for (int i = 0; i < TESTS; i++) {
        if (make_test(i, net)) num++;
    }
    cout << num/(float)TESTS << endl;
}