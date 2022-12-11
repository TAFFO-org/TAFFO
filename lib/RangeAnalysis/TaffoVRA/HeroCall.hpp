#pragma once

#include "VRAnalyzer.hpp"
#include "llvm/Support/Debug.h"
#include <functional>
#include <unordered_map>

namespace taffo
{

struct HeroCall {


private:
  static const auto &getHeroFunctions()
  {
    using namespace taffo;
    static std::unordered_map<std::string, std::function<void(VRAnalyzer *, const llvm::CallBase *)>> HeroFunctions = {
        //MEMCPY
        {"hero_memcpy_dev2host", &VRAnalyzer::handleHeroMemCpy},
        {"__dev-hero_memcpy_dev2host", &VRAnalyzer::handleHeroMemCpy},

        {"hero_memcpy_host2dev", &VRAnalyzer::handleHeroMemCpy},
        {"__dev-hero_memcpy_host2dev", &VRAnalyzer::handleHeroMemCpy},

        {"__dev-hero_memcpy_host2dev_async", &VRAnalyzer::handleHeroMemCpy},
        {"hero_memcpy_host2dev_async", &VRAnalyzer::handleHeroMemCpy},

        {"__dev-hero_memcpy_dev2host_async", &VRAnalyzer::handleHeroMemCpy},
        {"hero_memcpy_dev2host_async", &VRAnalyzer::handleHeroMemCpy},

        //MALLOC
        {"__dev-hero_l1malloc", &VRAnalyzer::handleMallocCall},
        {"hero_l1malloc", &VRAnalyzer::handleMallocCall},


    };
    return HeroFunctions;
  }


public:
  static bool isHeroCallInstruction(const std::string &function)
  {
    const auto &heroFunctions = getHeroFunctions();
    return heroFunctions.count(function) == 1;
  }

  static std::function<void(taffo::VRAnalyzer *, const llvm::CallBase *)> handleHeroCall(const std::string &function)
  {

    return getHeroFunctions().at(function);
  }
};

} // namespace taffo