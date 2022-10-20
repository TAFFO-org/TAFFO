#include "ConnectedComponents.h"

namespace taffo {

ConnectedComponents::ConnectedComponents(const int NodeCount, const std::list<std::pair<int, int>> &Edges)
    : nodeCount{NodeCount}, edges{Edges}
{
  calculateConnectedComponents();
}

int ConnectedComponents::merge(int* parent, int x)
{
  if (parent[x] == x)
    return x;
  return merge(parent, parent[x]);
}

void ConnectedComponents::calculateConnectedComponents()
{
  int parent[nodeCount];
  for (int i = 0; i < nodeCount; i++) {
    parent[i] = i;
  }
  for (auto x : edges) {
    parent[merge(parent, x.first)] = merge(parent, x.second);
  }
  for (int i = 0; i < nodeCount; i++) {
    parent[i] = merge(parent, parent[i]);
  }
  for (int i = 0; i < nodeCount; i++) {
    cc[parent[i]].push_back(i);
  }
}

}