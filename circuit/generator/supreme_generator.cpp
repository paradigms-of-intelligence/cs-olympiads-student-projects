#include <iostream>
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <array>


constexpr int INPUT_NODES = 784, OUTPUT_NODES = 1000;
int id = INPUT_NODES;
std::array<int, 4> layers = {5000, 3000, 1700, OUTPUT_NODES};
std::vector<std::pair<int, int>> edges;

std::map<int, std::map<int, int>> somemap;
int getnode(int a, int b) {
  if(a > b) std::swap(a, b);
  if(somemap[a][b] == 0) {
    ++id;
    edges.push_back({a, b});
    return somemap[a][b] = id;
  }
  return somemap[a][b];
}
int ID(int x, int y) {
  return 1+x+28*y;
}

int main() {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::vector<int> final;
  int test = 0;
  for(int x = 0; x < 25; ++x) {
  for(int y = 0; y < 25; ++y) {
      ++test;
     int nodes[] = {
      getnode(ID(x, y), ID(x+1, y+1)), getnode(ID(x+1, y), ID(x, y+1)), getnode(ID(x+2, y), ID(x+3, y+1)), getnode(ID(x+3, y), ID(x+2, y+1)),
      getnode(ID(x, y+2), ID(x+1, y+3)), getnode(ID(x+1, y+2), ID(x, y+3)), getnode(ID(x+2, y+2), ID(x+3, y+3)), getnode(ID(x+3, y+2), ID(x+2, y+3))
    };
    std::vector<int> newnodes(28);
    int idx = 0;
    for(int i = 0; i < 8; ++i) for(int j = 0; j < i; ++j) {
      newnodes[idx++] = getnode(nodes[i], nodes[j]);
    }
    std::shuffle(newnodes.begin(), newnodes.end(), rng);
    for(int i = 0; i < 7; ++i) {
      final.push_back(getnode(getnode(newnodes[4*i], newnodes[4*i+1]), getnode(newnodes[4*i+2], newnodes[4*i+3])));
    }
  }}
  std::cerr << test << ' ' << final.size() << ' ' << id << std::endl;
  for(int i = 0; i < layers.size(); ++i) {
    std::vector<int> nl = final, nr;
    std::shuffle(nl.begin(), nl.end(), rng);
    std::uniform_int_distribution<> dist(0, final.size()-1);
    while(nl.size() < layers[i]) {
      nl.push_back(final[dist(rng)]);
    }
    while(nr.size() < layers[i]) {
      nr.push_back(final[dist(rng)]);
    }
    final.clear();
    for(int j = 0; j < layers[i]; ++j) {
      final.push_back(getnode(nl[j], nr[j]));
    }
  }
  std::cout << id << '\n';
  for(auto [a, b] : edges) std::cout << a << ' ' << b << '\n';
  std::cout << OUTPUT_NODES << std::endl;
}
