#include <bits/stdc++.h>
#include "../circuit.h"
using namespace std;

struct AndNot_network {
    int N, O;
    vector<bool> value; // bit values for calculation
    vector<int> C_1, C_2; // edges
    vector<int> result_nodes; // ordered-ids of the final network output nodes

    void init(const char* path) {
        // initialize current network
        ifstream graphinput(path, ios::binary | ios::in);
        N = read_int32_t(graphinput);
        value.resize(N+1); C_1.resize(N+1); C_2.resize(N+1);

        for (int i = INPUT_NODES+1; i <= (int)N; i++) {
            int id = read_int32_t(graphinput);
            C_1[id] = read_int32_t(graphinput);
            C_2[id] = read_int32_t(graphinput);
        }
         
        O = read_int32_t(graphinput);
        result_nodes.resize(O);
        for (int i = 0; i < O; i++) result_nodes[i] = read_int32_t(graphinput);

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
        else if (abs(id) == ALWAYS_TRUE) c = 1;
        else if (abs(id) == ALWAYS_FALSE) c = 0;
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

    float guess(int correct) {
        // return the network's guess for MNIST input
        float g = 0;
        float cnt = 0;
        int category = O/10; // 10 categories for MNIST
        for (int counter = 0; counter < 10; counter++) {
            int on = 0;
            for (int k = 0; k < category; k++) {
                int id = result_nodes[counter * category + k];
                if (value[id])  on++;
            }
            if (on > category/2) {
                if(counter == correct) g = 1;
                cnt++;
            }
        }
        if(cnt == 0) return 0.1;

        return g/cnt;
    }
};

float make_test(AndNot_network &net, ifstream &in) {
    // 1 if test correct, 0 if not
    //to fix, variable `n` is useless
    vector<bool> input_values(INPUT_NODES);
    string input_image;
    getline(in, input_image);

    for (int i = 0; i < (int)INPUT_NODES; i++) {
        input_values[i] = (input_image[i] == '1');
    }

    net.input_into(input_values);

    net.calculatenetwork();

    string correct;
    getline(in, correct);
    return net.guess(correct[0]-'0');
}   

int main(int argc, char const *argv[]) {

    if(argc != 3) 
        program_abort(-1);

    // create the network structure
    AndNot_network net;
    net.init(argv[2]);


    ifstream test_input(argv[1], ios::in);
    // test on the test data
    float num = 0;
    for (int i = 0; i < TESTS; i++) {
        if(i%500 == 0) cout << "Tested " << i << "/" << TESTS << "\n";
        num += make_test(net, test_input);
    }
    
    cout << "Tested " << TESTS << "/" << TESTS << "\n";

    cout << 100.0 * num/(float)TESTS << "% accuracy\n";
}