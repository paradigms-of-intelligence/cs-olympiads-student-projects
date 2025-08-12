#include <iterator>
#include <iostream>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace OPTIONS {
  int bin_n = 20000, bin_o = 1000;
  double bin_d = .5;
  int uni_m = 2, uni_t = 1;
}

constexpr int INPUT_AMOUNT = 28*28;
int node_count = INPUT_AMOUNT;

std::random_device dev;
std::mt19937 rng(dev());

std::vector<std::pair<int, int>> conn;
std::set<int> unused;

int main() {
  std::cout << "bin_n: " << std::flush;
  std::cin >> OPTIONS::bin_n;
  std::cout << "bin_o: " << std::flush;
  std::cin >> OPTIONS::bin_o;
  std::cout << "bin_d: " << std::flush;
  std::cin >> OPTIONS::bin_d;
  std::cout << "uni_m: " << std::flush;
  std::cin >> OPTIONS::uni_m;
  std::cout << "uni_t: " << std::flush;
  std::cin >> OPTIONS::uni_t;
  for(int i = 1; i <= INPUT_AMOUNT; ++i) unused.insert(i);
  conn.assign(INPUT_AMOUNT, {-1, -1});
  std::binomial_distribution<> bindist(OPTIONS::bin_n, OPTIONS::bin_d);
  std::uniform_int_distribution<> uniform(1, OPTIONS::uni_m);
  int endsize = bindist(rng);
  while(node_count < OPTIONS::bin_o+INPUT_AMOUNT+endsize) {
    std::binomial_distribution<> bin3(node_count*2, .5);
    int link1 = 0;
    if(uniform(rng) <= OPTIONS::uni_t) {
      std::binomial_distribution<> bin2(((int) unused.size())*2, .5);
      int in1 = 0;
      while(in1 == 0) in1 = std::abs(bin2(rng)-(int) unused.size());
      auto it = std::begin(unused);
      std::advance(it, in1-1);
      link1 = *it;
      unused.erase(it);
    } else {
      while(link1 == 0) link1 = std::abs(bin3(rng)-node_count);
    }
    int link2 = 0;
    while(link2 == link1 || link2 == 0) link2 = std::abs(bin3(rng)-node_count);
    unused.erase(link2);
    conn.push_back({link1, link2});
    unused.insert(++node_count);
  }
  while(unused.empty() || unused.size()%10 != 0) {
    std::binomial_distribution<> bin(node_count*2, .5);
    int link1 = 0, link2 = 0;
    while(link1 == 0 || unused.find(link1) != unused.end()) link1 = std::abs(bin(rng)-node_count);
    while(link2 == 0 || link2 == link1 || unused.find(link1) != unused.end()) link2 = std::abs(bin(rng)-node_count);
    conn.push_back({link1, link2});
    unused.insert(++node_count);
  }
  std::vector<int> off;
  std::cout << node_count << std::endl;
  off.push_back(0);
  for(int i = 1; i <= conn.size(); ++i) {
    if(unused.find(i) != unused.end()) {
      off.push_back(off.back()+1);
      continue;
    }
    off.push_back(off.back());
    auto [a1, a2] = conn[i-1];
    a1 = a1 == -1?a1:a1-off[a1];
    a2 = a2 == -1?a2:a2-off[a2];
    std::cout << a1 << ' ' << a2 << std::endl;
  }
  for(int i : unused) {
    auto [a1, a2] = conn[i-1];
    a1 = a1 == -1?a1:a1-off[a1];
    a2 = a2 == -1?a2:a2-off[a2];
    std::cout << a1 << ' ' << a2 << std::endl;
  }
  std::cout << unused.size() << std::endl;
}
