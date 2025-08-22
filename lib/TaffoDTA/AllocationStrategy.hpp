#pragma once
#include "TaffoInfo/ValueInfo.hpp"

#include <memory>

namespace taffo {

/* this is the core of the strategy pattern for each new strategy
 * you should create a new class that inherits from dataTypeAllocationStrategy
 * and implement the apply, merge and isMergeable methods.
 * (the apply method is the actual strategy that will be applied to each value
 * the isMergeable method decides if two values are mergeable and the merge methods
 * decides how to merge them)
 *
 * When implementing a new strategy you should:
 * - create a new class that inherits from dataTypeAllocationStrategy
 * - implement the apply, merge and isMergeable methods
 * - add a new entry in the strategyMap in DataTypeAllocationPass.cpp
 * - add a new entry in the DtaStrategyType enum in DTAConfig.hpp and a new entry in the DtaStrategy in DTAConfig.cpp */
class AllocationStrategy {
public:
  virtual ~AllocationStrategy() = default;
  virtual bool apply(std::shared_ptr<ScalarInfo>& scalarInfo, llvm::Value* value) = 0;
  virtual bool isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                           std::shared_ptr<NumericTypeInfo> userNumericType) = 0;
  virtual std::shared_ptr<NumericTypeInfo> merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                                 const std::shared_ptr<NumericTypeInfo>& fpu) = 0;
};

// *** STRATEGIES DECLARATIONS ***
class FixedPointOnlyStrategy : public AllocationStrategy {
public:
  bool apply(std::shared_ptr<ScalarInfo>& scalarInfo, llvm::Value* value) override;
  bool isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                   std::shared_ptr<NumericTypeInfo> userNumericType) override;
  std::shared_ptr<NumericTypeInfo> merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                         const std::shared_ptr<NumericTypeInfo>& fpu) override;
};

class FloatingPointOnlyStrategy : public AllocationStrategy {
public:
  bool apply(std::shared_ptr<ScalarInfo>& scalarInfo, llvm::Value* value) override;
  bool isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                   std::shared_ptr<NumericTypeInfo> userNumericType) override;
  std::shared_ptr<NumericTypeInfo> merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                         const std::shared_ptr<NumericTypeInfo>& fpu) override;
};

class FixedFloatingPointStrategy : public AllocationStrategy {
public:
  bool apply(std::shared_ptr<ScalarInfo>& scalarInfo, llvm::Value* value) override;
  bool isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                   std::shared_ptr<NumericTypeInfo> userNumericType) override;
  std::shared_ptr<NumericTypeInfo> merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                         const std::shared_ptr<NumericTypeInfo>& fpu) override;
};

} // namespace taffo
