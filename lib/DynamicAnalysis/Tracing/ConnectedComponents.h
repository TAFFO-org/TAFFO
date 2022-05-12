#ifndef TAFFO_CONNECTEDCOMPONENTS_H
#define TAFFO_CONNECTEDCOMPONENTS_H

#include <list>
#include <unordered_map>
#include <memory>

namespace taffo
{

class ConnectedComponents
{
public:
  ConnectedComponents(int NodeCount, const std::list<std::pair<int, int>> &Edges);

  const std::unordered_map<int, std::list<int>>& getResult() {
      return cc;
  };

private:
  const int nodeCount;
  const std::list<std::pair<int, int>> &edges;
  std::unordered_map<int, std::list<int>> cc;

  void calculateConnectedComponents();
  int merge(int* parent, int x);
};

} // namespace taffo

#endif // TAFFO_CONNECTEDCOMPONENTS_H
