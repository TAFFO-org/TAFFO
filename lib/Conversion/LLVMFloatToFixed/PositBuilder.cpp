#include "LLVMFloatToFixedPass.h"
#include "PositBuilder.h"
#include "PositBuilderSoftware.h"
#include "PositBuilderRISCV.h"

using namespace flttofix;

std::unique_ptr<PositBuilder> PositBuilder::get(FloatToFixed *pass, llvm::IRBuilderBase &builder, const FixedPointType &metadata) {
    if (UseRiscvPPU) {
        return std::make_unique<PositBuilderRISCV>(pass, builder, metadata);
    } else {
        return std::make_unique<PositBuilderSoftware>(pass, builder, metadata);
    }
}
