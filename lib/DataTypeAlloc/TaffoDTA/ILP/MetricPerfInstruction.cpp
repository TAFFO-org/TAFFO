#include "MetricBase.h"
#include "Optimizer.h"
#include "Utils.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mdutils;
using namespace tuner;


static void emitError(const string &stringhina)
{
  LLVM_DEBUG(dbgs() << "[ERROR] " << stringhina << "\n");
}


void MetricPerf::handleDisabled(std::shared_ptr<tuner::OptimizerScalarInfo> res, const tuner::CPUCosts &cpuCosts, const char *start)
{
  auto constraint = vector<pair<string, double>>();
  auto &model = getModel();
  for (const auto &tmpString : cpuCosts.CostsIdValues) {
    if (tmpString.find(start) == 0 && cpuCosts.isDisabled(cpuCosts.decodeId(tmpString))) {

      if (hasDouble && tmpString.find("DOUBLE") != string::npos) {
        constraint.clear();
        constraint.push_back(make_pair(res->getDoubleSelectedVariable(), 1.0));
        model.insertLinearConstraint(constraint, Model::EQ, 0 /*, "Disable Double"*/);
      }

      if (hasHalf && tmpString.find("HALF") != string::npos) {
        constraint.clear();
        constraint.push_back(make_pair(res->getHalfSelectedVariable(), 1.0));
        model.insertLinearConstraint(constraint, Model::EQ, 0 /*,
                                      "Disable Half"*/
        );
      }

      if (hasQuad && tmpString.find("QUAD") != string::npos) {
        constraint.clear();
        constraint.push_back(make_pair(res->getQuadSelectedVariable(), 1.0));
        model.insertLinearConstraint(constraint, Model::EQ, 0 /*,
                                      "Disable Quad"*/
        );
      }

      if (hasPPC128 && tmpString.find("PPC128") != string::npos) {
        constraint.clear();
        constraint.push_back(make_pair(res->getPPC128SelectedVariable(), 1.0));
        model.insertLinearConstraint(constraint, Model::EQ, 0 /*,
                                      "Disable PPC128"*/
        );
      }

      if (hasFP80 && tmpString.find("FP80") != string::npos) {
        constraint.clear();
        constraint.push_back(make_pair(res->getFP80SelectedVariable(), 1.0));
        model.insertLinearConstraint(constraint, Model::EQ, 0 /*,
                                      "Disable FP80"*/
        );
      }

      if (hasBF16 && tmpString.find("BF16") != string::npos) {
        constraint.clear();
        constraint.push_back(make_pair(res->getBF16SelectedVariable(), 1.0));
        model.insertLinearConstraint(constraint, Model::EQ, 0 /*,
                                      "Disable BF16"*/
        );
      }
    }
  }
}


void MetricPerf::handleFAdd(BinaryOperator *instr, const unsigned OpCode, const shared_ptr<ValueInfo> &valueInfos)
{
  assert(instr->getOpcode() == llvm::Instruction::FAdd && "Operand mismatch!");

  auto &cpuCosts = getCpuCosts();
  auto &model = getModel();

  auto op1 = instr->getOperand(0);
  auto op2 = instr->getOperand(1);


  auto info1 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op1, instr));
  auto info2 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op2, instr));


  auto res = handleBinOpCommon(instr, op1, op2, true, valueInfos);
  if (!res)
    return;

  double maxCost = getCpuCosts().MaxMinCosts("ADD").first;

  handleDisabled(res, cpuCosts, "ADD");

  model.insertObjectiveElement(
      make_pair(res->getFixedSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_FIX)),
      MODEL_OBJ_MATHCOST, maxCost);
  model.insertObjectiveElement(
      make_pair(res->getFloatSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_FLOAT)),
      MODEL_OBJ_MATHCOST, 0);
  if (hasDouble) {
    model.insertObjectiveElement(
        make_pair(res->getDoubleSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_DOUBLE)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasHalf) {
    model.insertObjectiveElement(
        make_pair(res->getHalfSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_HALF)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasQuad) {
    model.insertObjectiveElement(
        make_pair(res->getQuadSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_QUAD)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasPPC128) {
    model.insertObjectiveElement(
        make_pair(res->getPPC128SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_PPC128)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasFP80) {
    model.insertObjectiveElement(
        make_pair(res->getFP80SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_FP80)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasBF16) {
    model.insertObjectiveElement(
        make_pair(res->getBF16SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::ADD_BF16)),
        MODEL_OBJ_MATHCOST, 0);
  }

  // Precision cost handled when allocating variable
  auto constraint = vector<pair<string, double>>();
  // Enob constraints
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info1->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, -1 /*, "Enob propagation in sum first addend"*/);

  // Enob constraints
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info2->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, -1 /*, "Enob propagation in sum second addend"*/);

  saveInfoForValue(instr, res);
}


void MetricPerf::handleFNeg(UnaryOperator *instr, const unsigned OpCode, const shared_ptr<ValueInfo> &valueInfos)
{
  assert(instr->getOpcode() == llvm::Instruction::FNeg && "Operand mismatch!");

  auto &cpuCosts = getCpuCosts();
  auto &model = getModel();

  auto op1 = instr->getOperand(0);

  auto info1 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op1, instr));

  /* adds type cast constraints for operands and returns the variable set of this instruction */
  auto res = handleUnaryOpCommon(instr, op1, true, valueInfos);
  if (!res)
    return;

  handleDisabled(res, cpuCosts, "SUB");

  double maxCost = getCpuCosts().MaxMinCosts("SUB").first;
  model.insertObjectiveElement(
      make_pair(res->getFixedSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_FIX)),
      MODEL_OBJ_MATHCOST, maxCost);
  model.insertObjectiveElement(
      make_pair(res->getFloatSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_FLOAT)),
      MODEL_OBJ_MATHCOST, 0);
  if (hasDouble) {
    model.insertObjectiveElement(
        make_pair(res->getDoubleSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_DOUBLE)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasHalf) {
    model.insertObjectiveElement(
        make_pair(res->getHalfSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_HALF)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasQuad) {
    model.insertObjectiveElement(
        make_pair(res->getQuadSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_QUAD)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasPPC128) {
    model.insertObjectiveElement(
        make_pair(res->getPPC128SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_PPC128)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasFP80) {
    model.insertObjectiveElement(
        make_pair(res->getFP80SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_FP80)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasBF16) {
    model.insertObjectiveElement(
        make_pair(res->getBF16SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_BF16)),
        MODEL_OBJ_MATHCOST, 0);
  }

  // Precision cost handled when allocating variable
  auto constraint = vector<pair<string, double>>();
  // Enob constraints
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info1->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, 0 /*, "Enob propagation in sub first addend"*/);

  saveInfoForValue(instr, res);
}


void MetricPerf::handleFSub(BinaryOperator *instr, const unsigned OpCode, const shared_ptr<ValueInfo> &valueInfos)
{
  assert(instr->getOpcode() == llvm::Instruction::FSub && "Operand mismatch!");

  auto &cpuCosts = getCpuCosts();
  auto &model = getModel();

  auto op1 = instr->getOperand(0);
  auto op2 = instr->getOperand(1);

  auto info1 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op1, instr));
  auto info2 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op2, instr));

  auto res = handleBinOpCommon(instr, op1, op2, true, valueInfos);
  if (!res)
    return;

  handleDisabled(res, cpuCosts, "SUB");

  double maxCost = getCpuCosts().MaxMinCosts("SUB").first;
  model.insertObjectiveElement(
      make_pair(res->getFixedSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_FIX)),
      MODEL_OBJ_MATHCOST, maxCost);
  model.insertObjectiveElement(
      make_pair(res->getFloatSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_FLOAT)),
      MODEL_OBJ_MATHCOST, 0);
  if (hasDouble) {
    model.insertObjectiveElement(
        make_pair(res->getDoubleSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_DOUBLE)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasHalf) {
    model.insertObjectiveElement(
        make_pair(res->getHalfSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_HALF)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasQuad) {
    model.insertObjectiveElement(
        make_pair(res->getQuadSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_QUAD)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasPPC128) {
    model.insertObjectiveElement(
        make_pair(res->getPPC128SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_PPC128)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasFP80) {
    model.insertObjectiveElement(
        make_pair(res->getFP80SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_FP80)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasBF16) {
    model.insertObjectiveElement(
        make_pair(res->getBF16SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::SUB_BF16)),
        MODEL_OBJ_MATHCOST, 0);
  }
  
  // Precision cost handled when allocating variable
  auto constraint = vector<pair<string, double>>();
  // Enob constraints
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info1->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, -1 /*, "Enob propagation in sub first addend"*/);

  // Enob constraints
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info2->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, -1 /*, "Enob propagation in sub second addend"*/);

  saveInfoForValue(instr, res);
}

void MetricPerf::handleFMul(BinaryOperator *instr, const unsigned OpCode, const shared_ptr<ValueInfo> &valueInfos)
{
  assert(instr->getOpcode() == llvm::Instruction::FMul && "Operand mismatch!");

  auto &cpuCosts = getCpuCosts();
  auto &model = getModel();

  auto op1 = instr->getOperand(0);
  auto op2 = instr->getOperand(1);

  auto info1 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op1, instr));
  auto info2 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op2, instr));

  auto res = handleBinOpCommon(instr, op1, op2, false, valueInfos);
  if (!res)
    return;
  handleDisabled(res, cpuCosts, "MUL");
  double maxCost = getCpuCosts().MaxMinCosts("MUL").first;

  model.insertObjectiveElement(
      make_pair(res->getFixedSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_FIX)),
      MODEL_OBJ_MATHCOST, maxCost);
  model.insertObjectiveElement(
      make_pair(res->getFloatSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_FLOAT)),
      MODEL_OBJ_MATHCOST, 0);
  if (hasDouble) {
    model.insertObjectiveElement(
        make_pair(res->getDoubleSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_DOUBLE)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasHalf) {
    model.insertObjectiveElement(
        make_pair(res->getHalfSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_HALF)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasQuad) {
    model.insertObjectiveElement(
        make_pair(res->getQuadSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_QUAD)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasPPC128) {
    model.insertObjectiveElement(
        make_pair(res->getPPC128SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_PPC128)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasFP80) {
    model.insertObjectiveElement(
        make_pair(res->getFP80SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_FP80)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasBF16) {
    model.insertObjectiveElement(
        make_pair(res->getBF16SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::MUL_BF16)),
        MODEL_OBJ_MATHCOST, 0);
  }

  // Precision cost handled when allocating variable
  int intbit_1 = getMaxIntBitOfValue(op1);
  int intbit_2 = getMaxIntBitOfValue(op2);

  auto constraint = vector<pair<string, double>>();
  // Enob constraint
  // c <= a + b - intbit_a - a
  // That is
  // c <= b - intbit_a
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info2->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, -intbit_1);

  // c <= a - intbit_b
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info1->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, -intbit_2);

  saveInfoForValue(instr, res);
}

void MetricPerf::handleFDiv(BinaryOperator *instr, const unsigned OpCode, const shared_ptr<ValueInfo> &valueInfos)
{
  assert(instr->getOpcode() == llvm::Instruction::FDiv && "Operand mismatch!");

  auto &cpuCosts = getCpuCosts();
  auto &model = getModel();

  auto op1 = instr->getOperand(0);
  auto op2 = instr->getOperand(1);

  auto info1 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op1, instr));
  auto info2 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(getInfoOfValue(op2, instr));

  auto res = handleBinOpCommon(instr, op1, op2, false, valueInfos);
  if (!res)
    return;

  double maxCost = getCpuCosts().MaxMinCosts("DIV").first;
  handleDisabled(res, cpuCosts, "DIV");

  model.insertObjectiveElement(
      make_pair(res->getFixedSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_FIX)),
      MODEL_OBJ_MATHCOST, maxCost);
  model.insertObjectiveElement(
      make_pair(res->getFloatSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_FLOAT)),
      MODEL_OBJ_MATHCOST, 0);
  if (hasDouble) {
    model.insertObjectiveElement(
        make_pair(res->getDoubleSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_DOUBLE)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasHalf) {
    model.insertObjectiveElement(
        make_pair(res->getHalfSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_HALF)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasQuad) {
    model.insertObjectiveElement(
        make_pair(res->getQuadSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_QUAD)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasPPC128) {
    model.insertObjectiveElement(
        make_pair(res->getPPC128SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_PPC128)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasFP80) {
    model.insertObjectiveElement(
        make_pair(res->getFP80SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_FP80)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasBF16) {
    model.insertObjectiveElement(
        make_pair(res->getBF16SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::DIV_BF16)),
        MODEL_OBJ_MATHCOST, 0);
  }

  // Precision cost handled when allocating variable
  auto constraint = vector<pair<string, double>>();

  int intbit_1 = getMaxIntBitOfValue(op1);
  int intbit_2 = getMaxIntBitOfValue(op2);
  int minbits2 = getMinIntBitOfValue(op2);

  // Enob constraint
  // Consider c = a / b, we have
  // c <= a + 2 * minbits_b + b - (intbit_a + a)
  // That is
  // c <= 2 * minbits_b + b - intbit_a
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info2->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, 2 * minbits2 - intbit_1);

  // c <= a + 2 * minbits_b + b - (intbit_b + b)
  // c <= a + 2 * minbits_b - intbit_b
  constraint.clear();
  constraint.push_back(make_pair(res->getRealEnobVariable(), 1.0));
  constraint.push_back(make_pair(info1->getRealEnobVariable(), -1.0));
  model.insertLinearConstraint(constraint, Model::LE, 2 * minbits2 - intbit_2);

  saveInfoForValue(instr, res);
}

void MetricPerf::handleFRem(BinaryOperator *instr, const unsigned OpCode, const shared_ptr<ValueInfo> &valueInfos)
{
  assert(instr->getOpcode() == llvm::Instruction::FRem && "Operand mismatch!");

  auto &cpuCosts = getCpuCosts();
  auto &model = getModel();

  auto op1 = instr->getOperand(0);
  auto op2 = instr->getOperand(1);

  auto res = handleBinOpCommon(instr, op1, op2, false, valueInfos);

  if (!res)
    return;

  double maxCost = getCpuCosts().MaxMinCosts("REM").first;
  handleDisabled(res, cpuCosts, "REM");

  model.insertObjectiveElement(
      make_pair(res->getFixedSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_FIX)),
      MODEL_OBJ_MATHCOST, maxCost);
  model.insertObjectiveElement(
      make_pair(res->getFloatSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_FLOAT)),
      MODEL_OBJ_MATHCOST, 0);
  if (hasDouble) {
    model.insertObjectiveElement(
        make_pair(res->getDoubleSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_DOUBLE)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasHalf) {
    model.insertObjectiveElement(
        make_pair(res->getHalfSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_HALF)),
        MODEL_OBJ_MATHCOST, 0);
  }
  if (hasQuad) {
    model.insertObjectiveElement(
        make_pair(res->getQuadSelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_QUAD)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasPPC128) {
    model.insertObjectiveElement(
        make_pair(res->getPPC128SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_PPC128)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasFP80) {
    model.insertObjectiveElement(
        make_pair(res->getFP80SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_FP80)),
        MODEL_OBJ_MATHCOST, 0);
  }

  if (hasBF16) {
    model.insertObjectiveElement(
        make_pair(res->getBF16SelectedVariable(), opt->getCurrentInstructionCost() * cpuCosts.getCost(CPUCosts::REM_BF16)),
        MODEL_OBJ_MATHCOST, 0);
  }

  saveInfoForValue(instr, res);

  // FIXME: insert enob propagation
  assert(false && "Enob propagation in frem not handled!");
}


void MetricPerf::handleCastInstruction(Instruction *instruction, shared_ptr<ValueInfo> valueInfo)
{
  LLVM_DEBUG(dbgs() << "Handling casting instruction...\n");

  if (isa<BitCastInst>(instruction)) {
    // FIXME: hack for jmeint to give info after the malloc
    auto bitcast = cast<BitCastInst>(instruction);

    if (bitcast->getType()->isPointerTy() && bitcast->getType()->getPointerElementType()->isFloatingPointTy()) {
      // When bitcasting to a floating point and having info, maybe we are dealing with a floating point array!
      auto fieldInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n");

        if (valueInfo->metadata) {
          LLVM_DEBUG(dbgs() << "WTF metadata has a value but it is not an input info...\n\n");
        } else {
          LLVM_DEBUG(dbgs() << "Metadata is really null.\n");
        }
        return;
      }

      auto fptype = dynamic_ptr_cast_or_null<FPType>(fieldInfo->IType);
      if (!fptype) {
        LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n");
        return;
      }

      // Do not save! As here we have a pointer!
      shared_ptr<OptimizerScalarInfo> variable = allocateNewVariableForValue(instruction, fptype, fieldInfo->IRange,
                                                                             fieldInfo->IError, false);

      auto met = make_shared<OptimizerPointerInfo>(variable);

      saveInfoForValue(instruction, met);

      LLVM_DEBUG(dbgs() << "Associated metadata " << met->toString() << " to the bitcast!\n");
      return;
    }

    LLVM_DEBUG(dbgs() << "[Warning] Bitcasting not supported for model generation.");
    return;
  }

  if (isa<FPExtInst>(instruction) ||
      isa<FPTruncInst>(instruction)) {
    handleFPPrecisionShift(instruction, valueInfo);
    return;
  }

  if (isa<TruncInst>(instruction) ||
      isa<ZExtInst>(instruction) ||
      isa<SExtInst>(instruction)) {
    LLVM_DEBUG(dbgs() << "Cast between integers, skipping...\n");
    return;
  }

  if (isa<UIToFPInst>(instruction) ||
      isa<SIToFPInst>(instruction)) {

    auto fieldInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata);
    if (!fieldInfo) {
      LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n");

      if (valueInfo->metadata) {
        LLVM_DEBUG(dbgs() << "WTF metadata has a value but it is not an input info...\n\n");
      } else {
        LLVM_DEBUG(dbgs() << "Metadata is really null.\n");
      }
      return;
    }

    auto fptype = dynamic_ptr_cast_or_null<FPType>(fieldInfo->IType);
    if (!fptype) {
      LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n");
      return;
    }

    shared_ptr<OptimizerScalarInfo> variable = allocateNewVariableForValue(instruction, fptype, fieldInfo->IRange,
                                                                           fieldInfo->IError);

    // Limiting the ENOB as coming from an integer we can have an error at min of 1
    // Look that here we have the original program, so these instruction are not related to fixed point implementation!
    auto constraint = vector<pair<string, double>>();
    constraint.clear();
    constraint.push_back(make_pair(variable->getRealEnobVariable(), 1.0));
    getModel().insertLinearConstraint(constraint, Model::LE, 1 /*, "Limiting Enob for integer to float conversion"*/);
    return;
  }

  if (isa<FPToSIInst>(instruction) ||
      isa<FPToUIInst>(instruction)) {
    LLVM_DEBUG(dbgs() << "Casting Floating point to integer, no costs introduced.\n");
    return;
  }

  if (isa<IntToPtrInst>(instruction) ||
      isa<PtrToIntInst>(instruction)) {
    LLVM_DEBUG(dbgs() << "Black magic with pointers is happening. We do not want to awake the dragon, rigth?\n");
    return;
  }

  llvm_unreachable("Did I really forgot something?");
}


void MetricPerf::handleStore(Instruction *instruction, const shared_ptr<ValueInfo> &valueInfo)
{
  auto *store = cast<StoreInst>(instruction);

  auto opWhereToStore = store->getPointerOperand();
  auto opRegister = store->getValueOperand();
  auto info2 = getInfoOfValue(opRegister, store);

  if (opRegister->getType()->isFloatingPointTy()) {
    auto info1 = getInfoOfValue(opWhereToStore, store);
    if (!info1 || !info2) {
      LLVM_DEBUG(dbgs() << "One of the two values does not have info, ignoring...\n";);
      return;
    }

    auto info_pointer_t = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(info1);
    if (!info_pointer_t) {
      emitError("No info on pointer value!");
      return;
    }

    auto info_variable_oeig_t = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info2);
    if (!info_variable_oeig_t) {
      emitError("No info on register value!");
      return;
    }

    LLVM_DEBUG(dbgs() << "Storing " << info2->toString() << " into " << info1->toString() << "\n");

    auto info_pointer = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info_pointer_t->getOptInfo());

    shared_ptr<OptimizerScalarInfo> variable = allocateNewVariableWithCastCost(opRegister, instruction);

    opt->insertTypeEqualityConstraint(info_pointer, variable, true);


    bool isConstant;

    if (!info_variable_oeig_t->doesReferToConstant()) {
      isConstant = false;
      // We do this only if storing a real result from a computation, if it comes from a constant we do not override the enob.
      // getModel().insertComment("Restriction for new enob [STORE]", 2);
      string newEnobVariable = info_pointer->getRealEnobVariable();
      newEnobVariable.append("_storeENOB");
      info_pointer->overrideEnob(newEnobVariable);
      // We force the enob back to the variable type, just in case!
      initRealEnobVariable(info_pointer);

      auto constraint = vector<pair<string, double>>();
      constraint.clear();
      constraint.push_back(make_pair(info_pointer->getRealEnobVariable(), 1.0));
      constraint.push_back(make_pair(info_variable_oeig_t->getRealEnobVariable(), -1.0));
      getModel().insertLinearConstraint(constraint, Model::LE, 0 /*, "Enob constraint ENOB propagation in load/store"*/);
    } else {
      LLVM_DEBUG(dbgs() << "[INFO] The value to store is a constant, not inserting it as may cause problems...\n");
      isConstant = true;
    }

    // We save the infos so we should retrieve them more quickly when using MemSSA
    // We save the ENOB of the stored variable that is the correct one to use
    auto a = make_shared<OptimizerScalarInfo>(info_variable_oeig_t->getBaseName(),
                                              info_variable_oeig_t->getMinBits(),
                                              info_pointer->getMaxBits(), info_pointer->getTotalBits(),
                                              info_pointer->isSigned,
                                              *info_pointer->getRange(), info_pointer->getOverridedEnob());
    a->setReferToConstant(isConstant);

    saveInfoForValue(instruction, a);


  } else if (opRegister->getType()->isPointerTy()) {
    // Storing a pointer. In the value there should be a pointer already, and the value where to store is, in fact,
    // a pointer to a pointer
    LLVM_DEBUG(dbgs() << "The register is: ";);
    LLVM_DEBUG(opRegister->print(dbgs()););
    LLVM_DEBUG(dbgs() << "\n";);
    LLVM_DEBUG(dbgs() << "Storing a pointer...\n";);
    if (!info2) {
      LLVM_DEBUG(dbgs() << "Skipping, as the value to save does not have any info...\n";);
      return;
    }

    if (info2->getKind() != OptimizerInfo::K_Pointer) {
      emitError("Storing to a pointer a value that is not a pointer, ouch!");
      return;
    }

    // We wrap the info with a dereference information
    // When a double load occours, for example, this will be handled succesfully, hopefully!
    saveInfoForPointer(opWhereToStore, make_shared<OptimizerPointerInfo>(info2));

  } else {
    LLVM_DEBUG(dbgs() << "Storing a non-floating point, skipping...\n";);
    return;
  }
}


void MetricPerf::handleFPPrecisionShift(Instruction *instruction, shared_ptr<ValueInfo> valueInfo)
{
  auto operand = instruction->getOperand(0); // The argument to be casted

  auto info = getInfoOfValue(operand, instruction);
  auto sinfos = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info);
  if (!sinfos) {
    LLVM_DEBUG(dbgs() << "No info for the operand, ignoring...\n");
    return;
  }

  // Copy information as for us is like a NOP
  saveInfoForValue(instruction, make_shared<OptimizerScalarInfo>(sinfos->getBaseName(),
                                                                 sinfos->getMinBits(),
                                                                 sinfos->getMaxBits(), sinfos->getTotalBits(),
                                                                 sinfos->isSigned, *sinfos->getRange(),
                                                                 sinfos->getOverridedEnob()));

  LLVM_DEBUG(dbgs() << "For this fpext/fptrunc, reusing variable" << sinfos->getBaseName() << "\n");
}

void MetricPerf::handlePhi(Instruction *instruction, shared_ptr<ValueInfo> valueInfo)
{
  auto *phi_n = cast<PHINode>(instruction);

  if (!phi_n->getType()->isFloatingPointTy()) {
    LLVM_DEBUG(dbgs() << "Phi node with non float value, skipping...\n");
    return;
  }

  // We can have two scenarios here: we can have value that came from a previous basic block, and therefore we have
  // already infos on it, or from a successor basic block, in this case we cannot have info about the value (e.g. loops)

  // In the former case we proceed as usual, in the latter case, we need to insert the value in a special set that will
  // be monitored in case of insertions. In that case, the phi loop can be closed.

  // We treat phi as normal assignment, without looking at the real "backend" implementation. This may be quite different
  // from the real execution, but the overall meaning is the same.

  assert(phi_n->getNumIncomingValues() >= 1
         && "Why on earth is there a Phi instruction with no incoming values?");

  auto fieldInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata);
  if (!fieldInfo) {
    LLVM_DEBUG(dbgs() << "Not metadata. Bailing out.\n\n");
    return;
  }

  if (!fieldInfo->IRange) {
    LLVM_DEBUG(dbgs() << "Not range information. Bailing out.\n\n");
    return;
  }

  auto fptype = dynamic_ptr_cast_or_null<FPType>(fieldInfo->IType);
  if (!fptype) {
    LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n\n");
    return;
  }

  // Allocating variable for result
  shared_ptr<OptimizerScalarInfo> variable =
    allocateNewVariableForValue(instruction, fptype, fieldInfo->IRange, fieldInfo->IError);

  auto &model = getModel();
  auto constraint = vector<pair<string, double>>();
  int missing = 0;
  for (unsigned index = 0; index < phi_n->getNumIncomingValues(); index++) {
    LLVM_DEBUG(dbgs() << "[Phi] Handling operand " << index << "...\n");
    Value *op = phi_n->getIncomingValue(index);

    if (auto info = getInfoOfValue(op, phi_n)) {
      if (auto info2 = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info)) {
        if (info2->doesReferToConstant()) {
          LLVM_DEBUG(dbgs() << "[INFO] Skipping " << *op << " as it is a constant!\n");
          continue;
        }

        LLVM_DEBUG(dbgs() << "[Phi] We have infos, treating as usual.\n");
        constraint.clear();
        constraint.push_back(make_pair(variable->getRealEnobVariable(), 1.0));
        constraint.push_back(make_pair(info2->getRealEnobVariable(), -1.0));
        model.insertLinearConstraint(constraint, Model::LE, 0);
      } else {
        llvm_unreachable("Should be a scalar!");
      }
    } else {
      LLVM_DEBUG(dbgs() << "[Phi] No value available, inserting in delayed set.\n");
      openPhiLoop(phi_n, op);
      missing++;
    }
  }

  LLVM_DEBUG(dbgs() << "[Phi] Elaboration concluded. Missing " << missing << " values.\n");

  getPhiWatcher().dumpState();
}


void MetricPerf::handleLoad(Instruction *instruction, const shared_ptr<ValueInfo> &valueInfo)
{
  LLVM_DEBUG(llvm::dbgs() << "Handle Load\n");
  if (!valueInfo) {
    LLVM_DEBUG(dbgs() << "No value info, skipping...\n");
    return;
  }

  auto *load = cast<LoadInst>(instruction);
  auto loaded = load->getPointerOperand();
  shared_ptr<OptimizerInfo> infos = getInfoOfValue(loaded, load);

  auto pinfos = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(infos);
  if (!pinfos) {
    emitError("Loaded a variable with no information attached, or attached info not a Pointer type!");
    return;
  }

  auto &model = getModel();
  if (load->getType()->isFloatingPointTy()) {
    auto sinfos = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(pinfos->getOptInfo());
    if (!sinfos) {
      emitError("Loaded a variable with no information attached...");
      return;
    }
    // We are copying the infos, still using variable types and all, the only problem is the enob

    string newEnobVariable = sinfos->getBaseEnobVariable();
    newEnobVariable.append("_memphi_");
    newEnobVariable.append(load->getFunction()->getName().str());
    newEnobVariable.append("_");
    newEnobVariable.append(uniqueIDForValue(load));
    std::replace(newEnobVariable.begin(), newEnobVariable.end(), '.', '_');
    LLVM_DEBUG(dbgs() << "New enob for load: " << newEnobVariable << "\n");
    model.createVariable(newEnobVariable, -BIG_NUMBER, BIG_NUMBER);

    auto constraint = vector<pair<string, double>>();
    constraint.clear();
    constraint.push_back(make_pair(newEnobVariable, 1.0));
    constraint.push_back(make_pair(sinfos->getBaseEnobVariable(), -1.0));
    model.insertLinearConstraint(constraint, Model::LE, 0);

    auto a = make_shared<OptimizerScalarInfo>(sinfos->getBaseName(),
                                              sinfos->getMinBits(),
                                              sinfos->getMaxBits(), sinfos->getTotalBits(),
                                              sinfos->isSigned,
                                              *sinfos->getRange(), newEnobVariable);
    saveInfoForValue(instruction, a);

    // We are loading a floating point, which means we have its value in a register.
    // As we cannot cast anything during a load, the register will use the very same variable

    // Running MemorySSA to find Values from which the load can actually load
    MemorySSA &memssa = getTuner()->getAnalysis<MemorySSAWrapperPass>(*load->getFunction()).getMSSA();
    taffo::MemSSAUtils memssa_utils(memssa);
    SmallVectorImpl<Value *> &def_vals = memssa_utils.getDefiningValues(load);
    def_vals.push_back(load->getPointerOperand());

    assert(def_vals.size() > 0 && "Loading an undefined value?");

    /* This is the same as for phi nodes. In particular, when using the MemSSA usually the most important
     * instructions that defines a values are stores. In particular, when looking at a store, we can use the enob
     * given to that store to understand the enob propagation. This enob will not change during the computation as
     * usually every store is only touched once, differently from PHINodes */
    for (Value *op: def_vals) {
      auto store = dyn_cast_or_null<StoreInst>(op);
      if (!store) {
        // We skip the variable if it is not a store
        if (op) {
          LLVM_DEBUG(dbgs() << "[INFO] Skipping " << *op << " as it is NOT a store!\n");
        } else {
          LLVM_DEBUG(dbgs() << "[INFO] Skipping null def.\n");
        }
        continue;
      }
      if (auto info = getInfoOfValue(op, nullptr)) {
        if (auto sinfo = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info)) {
          if (sinfo->doesReferToConstant()) {
            // We skip the variable if it is a constant
            LLVM_DEBUG(dbgs() << "[INFO] Skipping " << *op << " as it is a constant!\n");
            continue;
          }

          LLVM_DEBUG(dbgs() << "[memPhi] We have infos, treating as usual.\n");
          constraint.clear();
          constraint.push_back(make_pair(a->getRealEnobVariable(), 1.0));
          constraint.push_back(make_pair(sinfo->getRealEnobVariable(), -1.0));
          model.insertLinearConstraint(constraint, Model::LE, 0);
        } else {
          llvm_unreachable("Should be a scalar!");
        }
      } else {
        LLVM_DEBUG(dbgs() << "[Phi] No value available, inserting in delayed set.\n");
        openMemLoop(load, op);
      }
    }
  } else if (load->getType()->isPointerTy()) {
    LLVM_DEBUG(dbgs() << "Handling load of a pointer...\n";);
    // Unwrap the pointer, hoping that it is pointing to something
    auto info = pinfos->getOptInfo();
    if (info->getKind() != OptimizerInfo::K_Pointer) {
      LLVM_DEBUG(dbgs() << "Warning, returning a pointer but the unwrapped thing is not a pointer! To prevent error, wrapping it...";);
      // FIXME: hack to prevent problem when using global pointer as arrays
      LLVM_DEBUG(dbgs() << "Unfortunately got " << info->toString() << "\n";);

      info = make_shared<OptimizerPointerInfo>(info);
    }
    LLVM_DEBUG(dbgs() << "The final register will have as info: " << info->toString() << "\n";);
    saveInfoForValue(instruction, info);

  } else {
    LLVM_DEBUG(dbgs() << "Loading a non floating point value, ingoring.\n");
  }
}
