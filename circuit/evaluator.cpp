#include <bits/stdc++.h>
using namespace std;

const string networkinputfile = "network";
const string baseinputfile = "testdata/input_";
const int inputnodes = 784;
const int one = 2147483647, zero = 2147483646;
const int tests = 100;

// int reading from binary file
int readint(ifstream &stream) {
    char * buffer = new char [32];
    stream.read((char*)&buffer, 4);
    int r = strtol(buffer, &buffer, 2);
    delete[] buffer;
    cout << r << " DEGUB" << endl;
    return r;
}

struct network {
    int N;
    vector<bool> value; //bit values for calculation
    vector<int> C_1, C_2; //edges
    vector<int> resultnodes; // ordered-ids of the final network output nodes

    void init() {
        // initialize current network
        ifstream graphinput(networkinputfile, ios::binary);
        N = readint(graphinput)+1;
        value.resize(N);
        C_1.resize(N);
        C_2.resize(N);
        for (int i = 0; i < N; i++) {
            int id = readint(graphinput);
            C_1[id] = readint(graphinput);
            C_2[id] = readint(graphinput);
        }
        resultnodes.resize(10);
        for (int i = 0; i < 10; i++) resultnodes[i] = readint(graphinput);
        graphinput.close();
    }

    void input_into(vector<bool> &in) {
        for (int i = 1; i <= inputnodes; i++) value[i] = in[i-1];
    }

    int calculatenetwork() {
        // calculate all node values
        for (int i = inputnodes+1; i < N; i++) {
            auto getvalue = [&](int id) -> bool{
                bool c;
                if (id < 0) c = !value[id];
                else if (id == one) c = 1;
                else if (id == zero) c = 0;
                else c = value[id];
            };
            value[i] = getvalue(C_1[i]) & getvalue(C_2[i]);
        }
    }

    int guess() {
        // return the network's guess for MNIST
        int g = -1, c = 0;
        for (int o : resultnodes) {
            if (value[o]) {
                assert(g == -1);
                g = c;
            }
            c++;
        }
        assert(g != -1);
        return g;
    }
};

bool make_test(int n, network &net) {
    // 1 if test correct, 0 if not
    ifstream current_input(baseinputfile+(char)n+".txt");
    vector<bool> input_values;
    for (int i = 0; i < inputnodes; i++) {
        //get the input node values
    }
    //get the correct result and compare
}   


int main() {
    network net;
    net.init();
    float num = 0;
    for (int i = 0; i < tests; i++) {
        if (make_test(i, net)) num++;
    }
    cout << num/(float)tests << endl;
}