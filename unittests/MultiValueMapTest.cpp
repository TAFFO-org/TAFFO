#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"
#include "MultiValueMap.h"

namespace {

using namespace taffo;
using namespace llvm;


class MultiValueMapTest : public testing::Test {
protected:
  LLVMContext Context;
  Constant *ConstantV;
  std::unique_ptr<BitCastInst> BitcastV;
  std::unique_ptr<BinaryOperator> AddV;
  std::unique_ptr<ICmpInst> CmpV;
  std::unique_ptr<BinaryOperator> SubV;
    
  MultiValueMapTest() :
      ConstantV(ConstantInt::get(Type::getInt32Ty(Context), 0)),
      BitcastV(new BitCastInst(ConstantV, Type::getInt32Ty(Context))),
      AddV(BinaryOperator::CreateAdd(ConstantV, ConstantV)),
      CmpV(new ICmpInst(ICmpInst::Predicate::ICMP_EQ, ConstantV, ConstantV)),
      SubV(BinaryOperator::CreateSub(ConstantV, ConstantV)) {
    DebugFlag = true;
  }
      
  template <typename VMapT, typename ExpListT>
  static bool compareKeysToExpected(VMapT& VVMap, ExpListT& Expected) {
    for (auto EInner: Expected) {
      for (auto V: EInner) {
        SmallVector<Value *, 2> All;
        VVMap.getAssociatedValues(V, All);
        if (EInner != All)
          return false;
      }
    }
    return true;
  }
  
  template <typename ItT, typename ExpListT>
  static bool comparePairsFromItToExpected(ItT CI, ItT CEnd, ExpListT& Expected) {
    for (auto ExpectedPair: Expected) {
      if (CI == CEnd) return false;
      if (ExpectedPair.first != CI->first) return false;
      if (ExpectedPair.second != CI->second) return false;
      CI++;
    }
    if (CI != CEnd) return false;
    return true;
  }
};


TEST_F(MultiValueMapTest, Empty) {
  MultiValueMap<Value *, int> VVMap;
  ASSERT_EQ(VVMap.size(), 0);
  ASSERT_EQ(VVMap.empty(), true);
  ASSERT_EQ(VVMap.begin(), VVMap.end());
}

TEST_F(MultiValueMapTest, InsertOneWhenEmpty) {
  MultiValueMap<Value *, int> VVMap;
  
  auto R1 = VVMap.insert(VVMap.begin(), {this->AddV.get(), 10});
  ASSERT_EQ(R1.second, true);
  ASSERT_EQ(R1.first->first, this->AddV.get());
  ASSERT_EQ(R1.first->second, 10);
  ASSERT_EQ(R1.first, VVMap.begin());
  auto IShouldBeEnd = R1.first; IShouldBeEnd++;
  ASSERT_EQ(IShouldBeEnd, VVMap.end());
  
  auto R2 = VVMap.insert(VVMap.begin(), {this->AddV.get(), 20});
  ASSERT_EQ(R2.second, false);
  ASSERT_EQ(R2.first, VVMap.begin());
  ASSERT_EQ(R2.first, R1.first);
}

TEST_F(MultiValueMapTest, InsertOneAtBeginning) {
  MultiValueMap<Value *, int> VVMap;
  
  VVMap.insert(VVMap.begin(), {this->AddV.get(), 10});
  auto R1 = VVMap.insert(VVMap.begin(), {this->BitcastV.get(), 30});
  ASSERT_EQ(R1.second, true);
  ASSERT_EQ(R1.first->first, this->BitcastV.get());
  ASSERT_EQ(R1.first->second, 30);
  ASSERT_EQ(R1.first, VVMap.begin());
  auto I2 = ++(R1.first);
  ASSERT_EQ(I2->first, this->AddV.get());
  ASSERT_EQ(I2->second, 10);
  auto IShouldBeEnd = I2; IShouldBeEnd++;
  ASSERT_EQ(IShouldBeEnd, VVMap.end());
}

TEST_F(MultiValueMapTest, InsertOneAtEnd) {
  MultiValueMap<Value *, int> VVMap;
  
  auto R1 = VVMap.insert(VVMap.begin(), {this->AddV.get(), 10});
  auto R2 = VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  ASSERT_EQ(R2.second, true);
  ASSERT_EQ(R2.first->first, this->BitcastV.get());
  ASSERT_EQ(R2.first->second, 30);
  auto R2Dup = R1.first; R2Dup++;
  ASSERT_EQ(R2Dup, R2.first);
  auto IShouldBeEnd = R2.first; IShouldBeEnd++;
  ASSERT_EQ(IShouldBeEnd, VVMap.end());
}

TEST_F(MultiValueMapTest, InsertBulkPairs) {
  SmallVector<std::pair<Value *, int>, 3> Expected =
      {{this->AddV.get(), 10},
       {this->ConstantV, 20},
       {this->BitcastV.get(), 30}};
   
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.begin(), Expected.begin(), Expected.end());
  ASSERT_TRUE(comparePairsFromItToExpected(VVMap.begin(), VVMap.end(), Expected));
}

TEST_F(MultiValueMapTest, InsertBulkAssociated) {
  SmallVector<Value *, 3> Group =
      {this->AddV.get(),
       this->ConstantV,
       this->BitcastV.get()};
   
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.begin(), Group.begin(), Group.end(), 10);
  SmallVector<SmallVector<Value *, 3>, 2> Expected = {Group};
  ASSERT_TRUE(compareKeysToExpected(VVMap, Expected));
}

TEST_F(MultiValueMapTest, AssociateOneLeft) {
  MultiValueMap<Value *, int> VVMap;
  
  auto P1 = VVMap.insert(VVMap.begin(), {this->AddV.get(), 10});
  auto P2 = VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  auto P1K0 = VVMap.insertLeft(P1.first, this->ConstantV);
  ASSERT_EQ(P1K0.first, P1.first);
  ASSERT_EQ(P1K0.second, false);
  
  auto P1K1 = VVMap.insertLeft(P2.first, this->ConstantV);
  ASSERT_EQ(P1K1.second, true);
  ASSERT_EQ(P1K1.first->first, this->ConstantV);
  ASSERT_EQ(P1K1.first->second, 10);
  P2.first = P1K1.first; P2.first++;
  ASSERT_EQ(P2.first->first, this->BitcastV.get());
  ASSERT_EQ(P2.first->second, 30);
  
  auto P2K2 = VVMap.insertLeft(VVMap.end(), this->CmpV.get());
  ASSERT_EQ(P2K2.second, true);
  ASSERT_EQ(P2K2.first->first, this->CmpV.get());
  ASSERT_EQ(P2K2.first->second, 30);
  P2.first = P2K2.first; P2.first++;
  ASSERT_EQ(P2.first, VVMap.end());
  
  SmallVector<SmallVector<Value *, 2>, 2> Expected =
      {{this->AddV.get(), this->ConstantV},
       {this->BitcastV.get(), this->CmpV.get()}};
  ASSERT_TRUE(compareKeysToExpected(VVMap, Expected));
}

TEST_F(MultiValueMapTest, AssociateOneRight) {
  MultiValueMap<Value *, int> VVMap;
  
  auto P1 = VVMap.insert(VVMap.begin(), {this->AddV.get(), 10});
  auto P2 = VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  auto P1K1 = VVMap.insertRight(P1.first, this->ConstantV);
  ASSERT_EQ(P1K1.second, true);
  ASSERT_EQ(P1K1.first->first, this->ConstantV);
  ASSERT_EQ(P1K1.first->second, 10);
  P2.first = P1K1.first; P2.first++;
  ASSERT_EQ(P2.first->first, this->AddV.get());
  ASSERT_EQ(P2.first->second, 10);
  P2.first++;
  ASSERT_EQ(P2.first->first, this->BitcastV.get());
  ASSERT_EQ(P2.first->second, 30);
  
  auto P2K1 = VVMap.insertRight(P2.first, this->SubV.get());
  ASSERT_EQ(P2K1.second, true);
  ASSERT_EQ(P2K1.first->first, this->SubV.get());
  ASSERT_EQ(P2K1.first->second, 30);
  P2.first = P2K1.first; P2.first++;
  ASSERT_EQ(P2.first->first, this->BitcastV.get());
  ASSERT_EQ(P2.first->second, 30);
  
  auto P2K2 = VVMap.insertRight(VVMap.end(), this->CmpV.get());
  ASSERT_EQ(P2K2.first, VVMap.end());
  ASSERT_EQ(P2K2.second, false);
  
  SmallVector<SmallVector<Value *, 2>, 2> Expected =
      {{this->ConstantV, this->AddV.get()},
       {this->SubV.get(), this->BitcastV.get()}};
  ASSERT_TRUE(compareKeysToExpected(VVMap, Expected));
}

TEST_F(MultiValueMapTest, ConstIterator) {
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.end(), {this->AddV.get(), 10});
  VVMap.insert(VVMap.end(), {this->ConstantV, 20});
  VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  SmallVector<std::pair<Value *, int>, 3> Expected =
      {{this->AddV.get(), 10},
       {this->ConstantV, 20},
       {this->BitcastV.get(), 30}};
  const MultiValueMap<Value *, int>& CVVMap = VVMap;
  MultiValueMap<Value *, int>::const_iterator CI = CVVMap.begin();
  ASSERT_TRUE(comparePairsFromItToExpected(CI, CVVMap.end(), Expected));
}

TEST_F(MultiValueMapTest, LazyIterator) {
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.end(), {this->AddV.get(), 10});
  auto G1 = VVMap.insert(VVMap.end(), {this->ConstantV, 20});
  VVMap.insertLeft(++G1.first, this->SubV.get());
  VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  auto I1 = VVMap.find(this->ConstantV);
  SmallVector<std::pair<Value *, int>, 3> Expected =
      {{this->ConstantV, 20},
       {this->SubV.get(), 20},
       {this->BitcastV.get(), 30}};
  ASSERT_TRUE(comparePairsFromItToExpected(I1, VVMap.end(), Expected));
}

TEST_F(MultiValueMapTest, BackwardsLazyIterator) {
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.end(), {this->AddV.get(), 10});
  auto G1 = VVMap.insert(VVMap.end(), {this->ConstantV, 20});
  VVMap.insertLeft(++G1.first, this->SubV.get());
  VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  auto I = VVMap.find(this->SubV.get());
  SmallVector<std::pair<Value *, int>, 3> Expected =
      {{this->SubV.get(), 20},
       {this->ConstantV, 20},
       {this->AddV.get(), 10}};
  auto ExpI = Expected.begin();
  ASSERT_EQ(I->first, ExpI->first);
  ASSERT_EQ(I->second, ExpI->second);
  while (I != VVMap.begin()) {
    --I; ++ExpI;
    ASSERT_EQ(I->first, ExpI->first);
    ASSERT_EQ(I->second, ExpI->second);
  }
}

TEST_F(MultiValueMapTest, EraseOneFromList) {
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.end(), {this->AddV.get(), 10});
  auto G1 = VVMap.insert(VVMap.end(), {this->ConstantV, 20});
  VVMap.insertLeft(++G1.first, this->SubV.get());
  VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  bool R = VVMap.erase(this->ConstantV);
  ASSERT_EQ(R, true);
  SmallVector<std::pair<Value *, int>, 3> Exp =
      {{this->AddV.get(), 10},
       {this->SubV.get(), 20},
       {this->BitcastV.get(), 30}};
  ASSERT_TRUE(comparePairsFromItToExpected(VVMap.begin(), VVMap.end(), Exp));
}

TEST_F(MultiValueMapTest, EraseListOfOne) {
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.end(), {this->AddV.get(), 10});
  auto G1 = VVMap.insert(VVMap.end(), {this->ConstantV, 20});
  VVMap.insertLeft(++G1.first, this->SubV.get());
  VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  auto F = VVMap.find(this->AddV.get());
  ASSERT_NE(F, VVMap.end());
  F = VVMap.erase(F);
  ASSERT_EQ(F->first, this->ConstantV);
  ASSERT_EQ(F->second, 20);
  SmallVector<std::pair<Value *, int>, 3> Exp =
      {{this->ConstantV, 20},
       {this->SubV.get(), 20},
       {this->BitcastV.get(), 30}};
  ASSERT_TRUE(comparePairsFromItToExpected(VVMap.begin(), VVMap.end(), Exp));
}

TEST_F(MultiValueMapTest, EraseAll) {
  MultiValueMap<Value *, int> VVMap;
  VVMap.insert(VVMap.end(), {this->AddV.get(), 10});
  auto G1 = VVMap.insert(VVMap.end(), {this->ConstantV, 20});
  VVMap.insertLeft(++G1.first, this->SubV.get());
  VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  auto F = VVMap.find(this->SubV.get());
  ASSERT_NE(F, VVMap.end());
  F = VVMap.eraseAll(F);
  ASSERT_EQ(F->first, this->BitcastV.get());
  ASSERT_EQ(F->second, 30);
  SmallVector<std::pair<Value *, int>, 3> Exp =
      {{this->AddV.get(), 10},
       {this->BitcastV.get(), 30}};
  ASSERT_TRUE(comparePairsFromItToExpected(VVMap.begin(), VVMap.end(), Exp));
}

TEST_F(MultiValueMapTest, RAUWCallback) {
  struct MVMTestCfg: public MultiValueMapConfig<Value *> {
    struct ExtraData {
      Value *ChkOld;
      Value *ChkNew;
    };
    static void onRAUW(const ExtraData& Data, Value *OldK, Value *NewK) {
      ASSERT_EQ(Data.ChkOld, OldK);
      ASSERT_EQ(Data.ChkNew, NewK);
    }
    static void onDelete(const ExtraData& Data, Value *K) {
      GTEST_FATAL_FAILURE_("onDelete called, but the operation was RAUW");
    }
  };

  MVMTestCfg::ExtraData XData;
  XData.ChkOld = this->SubV.get();
  XData.ChkNew = ConstantV;
  
  MultiValueMap<Value *, int, MVMTestCfg> VVMap(XData);
  VVMap.insert(VVMap.end(), {this->AddV.get(), 10});
  VVMap.insert(VVMap.end(), {this->SubV.get(), 20});
  VVMap.insert(VVMap.end(), {this->BitcastV.get(), 30});
  
  this->SubV->replaceAllUsesWith(ConstantV);
  
  auto F = VVMap.find(ConstantV);
  ASSERT_NE(F, VVMap.end());
  ASSERT_EQ(F->first, ConstantV);
  ASSERT_EQ(F->second, 20);
}


}
