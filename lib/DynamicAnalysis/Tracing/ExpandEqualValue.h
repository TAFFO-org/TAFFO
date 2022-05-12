#ifndef TAFFO_EXPANDEQUALVALUE_H
#define TAFFO_EXPANDEQUALVALUE_H

#include <list>
#include <unordered_map>

#include "TracingUtils.h"

namespace taffo
{

class ExpandEqualValue
{
public:
  ExpandEqualValue(std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>>& CCValues);
  std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>>& getResult() {
    return expandedCCValues;
  }

private:
  std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>>& ccValues;
  std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>> expandedCCValues;

  void expandEqualValues();
};

}


#endif // TAFFO_EXPANDEQUALVALUE_H
