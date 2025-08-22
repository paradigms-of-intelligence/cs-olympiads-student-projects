#include "../circuit.h"
#include <bits/stdc++.h>
#include <cassert>
#include <cstdio>
using namespace std;

unsigned char getnoneofch(FILE *file) {
  int ch = getc(file);
  if (ch != EOF)
    return ch;

  fprintf(stderr, "*** decode: unexpected EOF\n");
  exit(1);
}

unsigned decode(FILE *file) {
  unsigned x = 0, i = 0;
  unsigned char ch;

  while ((ch = getnoneofch(file)) & 0x80)
    x |= (ch & 0x7f) << (7 * i++);

  return x | (ch << (7 * i));
}

struct AndNot_network {
  vector<char> value;       // bit values for calculation, as char b/c it is faster.
  vector<int> C_1, C_2;     // edges
  vector<int> result_nodes; // ordered-ids of the final network output nodes

  // Hey, I'm Linus Torvalds. I fixed this code, thank me later.

  void init(const char *path) {
    // initialize current network
    FILE *in = fopen(path, "r");
    int M, I, L, O, A;
    fscanf(in, "aig %d %d %d %d %d\n", &M, &I, &L, &O, &A);
    assert(I == INPUT_NODES);
    assert(L == 0);
    assert(O == OUTPUT_NODES);
    for (int i = 0; i < O; i++) {
      int v;
      fscanf(in, "%d\n", &v);
      result_nodes.push_back(v);
    }
    value.resize(M + 1);
    C_1.resize(M + 1);
    C_2.resize(M + 1);

    for (int i = INPUT_NODES + 1; i <= (int)M; i++) {
      int idx = 2 * i;
      int delta0 = decode(in);
      int delta1 = decode(in);
      int a = idx - delta0;
      int b = a - delta1;
      C_1[i] = a;
      C_2[i] = b;
    }

    fclose(in);
  }

  void input_into(vector<bool> &in) {
    for (int i = 1; i <= (int)INPUT_NODES; i++)
      value[i] = in[i - 1];
  }

  bool getvalue(int id) {
    int bit = value[id / 2];
    return id % 2 ? !bit : bit;
  }

  void calculatenetwork() {
    // calculate all node values
    for (int i = INPUT_NODES + 1; i < (int)value.size(); i++) {
      value[i] = getvalue(C_1[i]) & getvalue(C_2[i]);
    }
  }

  float guess(int correct) {
    // return the network's guess for MNIST input
    float g = 0;
    float cnt = 0;
    int category = OUTPUT_NODES / 10; // 10 categories for MNIST
    vector<pair<int, int>> category_ids(10, make_pair(0, 0));
    for (int counter = 0; counter < 10; counter++) {
      int on = 0;
      for (int k = 0; k < category; k++) {
        int id = result_nodes[counter * category + k];
        if (getvalue(id))
          on++;
      }
      category_ids[counter] = {on, counter};
    }
    sort(category_ids.begin(), category_ids.end(), greater<pair<int, int>>());

    int maximum = category_ids[0].first;
    int i = 0;
    while (category_ids[i].first == maximum && i < 10) {
      cnt++;
      if (category_ids[i].second == correct)
        g++;
      i++;
    }
    return g / cnt;
  }
};

float make_test(AndNot_network &net, ifstream &in) {
  // 1 if test correct, 0 if not
  // to fix, variable `n` is useless
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
  return net.guess(correct[0] - '0');
}

int main(int argc, char const *argv[]) {
  if (argc != 3)
    program_abort(-1);

  // create the network structure
  AndNot_network net;
  net.init(argv[2]);

  ifstream test_input(argv[1], ios::in);
  // test on the test data
  float num = 0;
  for (int i = 0; i < TESTS; i++) {
    if (i % 500 == 0)
      cout << "Tested " << i << "/" << TESTS << "\n";
    num += make_test(net, test_input);
  }

  cout << "Tested " << TESTS << "/" << TESTS << "\n";

  cout << 100.0 * num / (float)TESTS << "% accuracy\n";
}
