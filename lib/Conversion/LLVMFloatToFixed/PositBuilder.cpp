#include "LLVMFloatToFixedPass.h"
#include "PositBuilder.h"
#include "PositBuilderSoftware.h"

using namespace flttofix;

std::unique_ptr<PositBuilder> PositBuilder::get(FloatToFixed *pass, llvm::IRBuilderBase &builder, const FixedPointType &metadata) {
    if (UseRiscvPPU) {
        llvm_unreachable("Unimplemented!");
    } else {
        return std::make_unique<PositBuilderSoftware>(pass, builder, metadata);
    }
}
