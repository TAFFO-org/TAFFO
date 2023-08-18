#include "include/posit.h"

namespace posit {

#define P32 Posit<int32_t, 32, 2, uint32_t, PositSpec::WithNan>
#define P16 Posit<int16_t, 16, 2, uint16_t, PositSpec::WithNan>
#define P8 Posit<int8_t, 8, 2, uint8_t, PositSpec::WithNan>

template class P32;
template class P16;
template class P8;


/* ------------------------- Posit to Posit casts ------------------------- */
template P32 P16::to_posit();
template P32 P8::to_posit();
template P16 P32::to_posit();
template P16 P8::to_posit();
template P8 P32::to_posit();
template P8 P16::to_posit();


/* ------------------------------ Operators ------------------------------- */
#define EXPAND_OPS(PX) \
template bool operator==(const PX&, const PX&); \
template bool operator!=(const PX&, const PX&); \
template bool operator<=(const PX&, const PX&); \
template bool operator>=(const PX&, const PX&); \
template bool operator<(const PX&, const PX&); \
template bool operator>(const PX&, const PX&); \
template PX operator+(const PX&, const PX&); \
template PX operator-(const PX&, const PX&); \
template PX operator*(const PX&, const PX&); \
template PX operator/(const PX&, const PX&); \
template PX fma(const PX&, const PX&, const PX&); \

EXPAND_OPS(P32)
EXPAND_OPS(P16)
EXPAND_OPS(P8)


/* ------------------------ Fixed point conversions ----------------------- */
template <class PX, class fixed_t, int frac>
void from_fixed(PX *out, fixed_t src) {
    using BE=typename PX::BackendT;
    new (out) PX(BE::template make_fixed<fixedtrait<fixed_t, sizeof(fixed_t) * 8, frac>>(src));
}

template <class PX, class fixed_t, int frac>
fixed_t to_fixed(PX *src) {
    return src->to_backend().template pack_xfixed<fixedtrait<fixed_t, sizeof(fixed_t) * 8, frac>>();
}

#define EXPAND_FROM_FIXED(PX) \
template void from_fixed<PX, int8_t, 0>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 1>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 2>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 3>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 4>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 5>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 6>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 7>(PX *, int8_t); \
template void from_fixed<PX, int8_t, 8>(PX *, int8_t); \
\
template void from_fixed<PX, int16_t, 0>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 1>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 2>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 3>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 4>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 5>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 6>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 7>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 8>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 9>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 10>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 11>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 12>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 13>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 14>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 15>(PX *, int16_t); \
template void from_fixed<PX, int16_t, 16>(PX *, int16_t); \
\
template void from_fixed<PX, int32_t, 0>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 1>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 2>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 3>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 4>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 5>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 6>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 7>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 8>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 9>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 10>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 11>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 12>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 13>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 14>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 15>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 16>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 17>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 18>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 19>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 20>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 21>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 22>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 23>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 24>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 25>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 26>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 27>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 28>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 29>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 30>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 31>(PX *, int32_t); \
template void from_fixed<PX, int32_t, 32>(PX *, int32_t); \
\
template void from_fixed<PX, int64_t, 0>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 1>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 2>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 3>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 4>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 5>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 6>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 7>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 8>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 9>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 10>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 11>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 12>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 13>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 14>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 15>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 16>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 17>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 18>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 19>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 20>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 21>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 22>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 23>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 24>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 25>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 26>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 27>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 28>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 29>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 30>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 31>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 32>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 33>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 34>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 35>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 36>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 37>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 38>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 39>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 40>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 41>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 42>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 43>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 44>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 45>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 46>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 47>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 48>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 49>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 50>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 51>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 52>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 53>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 54>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 55>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 56>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 57>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 58>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 59>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 60>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 61>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 62>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 63>(PX *, int64_t); \
template void from_fixed<PX, int64_t, 64>(PX *, int64_t); \

#define EXPAND_TO_FIXED(PX) \
template int8_t to_fixed<PX, int8_t, 0>(PX *); \
template int8_t to_fixed<PX, int8_t, 1>(PX *); \
template int8_t to_fixed<PX, int8_t, 2>(PX *); \
template int8_t to_fixed<PX, int8_t, 3>(PX *); \
template int8_t to_fixed<PX, int8_t, 4>(PX *); \
template int8_t to_fixed<PX, int8_t, 5>(PX *); \
template int8_t to_fixed<PX, int8_t, 6>(PX *); \
template int8_t to_fixed<PX, int8_t, 7>(PX *); \
template int8_t to_fixed<PX, int8_t, 8>(PX *); \
\
template int16_t to_fixed<PX, int16_t, 0>(PX *); \
template int16_t to_fixed<PX, int16_t, 1>(PX *); \
template int16_t to_fixed<PX, int16_t, 2>(PX *); \
template int16_t to_fixed<PX, int16_t, 3>(PX *); \
template int16_t to_fixed<PX, int16_t, 4>(PX *); \
template int16_t to_fixed<PX, int16_t, 5>(PX *); \
template int16_t to_fixed<PX, int16_t, 6>(PX *); \
template int16_t to_fixed<PX, int16_t, 7>(PX *); \
template int16_t to_fixed<PX, int16_t, 8>(PX *); \
template int16_t to_fixed<PX, int16_t, 9>(PX *); \
template int16_t to_fixed<PX, int16_t, 10>(PX *); \
template int16_t to_fixed<PX, int16_t, 11>(PX *); \
template int16_t to_fixed<PX, int16_t, 12>(PX *); \
template int16_t to_fixed<PX, int16_t, 13>(PX *); \
template int16_t to_fixed<PX, int16_t, 14>(PX *); \
template int16_t to_fixed<PX, int16_t, 15>(PX *); \
template int16_t to_fixed<PX, int16_t, 16>(PX *); \
\
template int32_t to_fixed<PX, int32_t, 0>(PX *); \
template int32_t to_fixed<PX, int32_t, 1>(PX *); \
template int32_t to_fixed<PX, int32_t, 2>(PX *); \
template int32_t to_fixed<PX, int32_t, 3>(PX *); \
template int32_t to_fixed<PX, int32_t, 4>(PX *); \
template int32_t to_fixed<PX, int32_t, 5>(PX *); \
template int32_t to_fixed<PX, int32_t, 6>(PX *); \
template int32_t to_fixed<PX, int32_t, 7>(PX *); \
template int32_t to_fixed<PX, int32_t, 8>(PX *); \
template int32_t to_fixed<PX, int32_t, 9>(PX *); \
template int32_t to_fixed<PX, int32_t, 10>(PX *); \
template int32_t to_fixed<PX, int32_t, 11>(PX *); \
template int32_t to_fixed<PX, int32_t, 12>(PX *); \
template int32_t to_fixed<PX, int32_t, 13>(PX *); \
template int32_t to_fixed<PX, int32_t, 14>(PX *); \
template int32_t to_fixed<PX, int32_t, 15>(PX *); \
template int32_t to_fixed<PX, int32_t, 16>(PX *); \
template int32_t to_fixed<PX, int32_t, 17>(PX *); \
template int32_t to_fixed<PX, int32_t, 18>(PX *); \
template int32_t to_fixed<PX, int32_t, 19>(PX *); \
template int32_t to_fixed<PX, int32_t, 20>(PX *); \
template int32_t to_fixed<PX, int32_t, 21>(PX *); \
template int32_t to_fixed<PX, int32_t, 22>(PX *); \
template int32_t to_fixed<PX, int32_t, 23>(PX *); \
template int32_t to_fixed<PX, int32_t, 24>(PX *); \
template int32_t to_fixed<PX, int32_t, 25>(PX *); \
template int32_t to_fixed<PX, int32_t, 26>(PX *); \
template int32_t to_fixed<PX, int32_t, 27>(PX *); \
template int32_t to_fixed<PX, int32_t, 28>(PX *); \
template int32_t to_fixed<PX, int32_t, 29>(PX *); \
template int32_t to_fixed<PX, int32_t, 30>(PX *); \
template int32_t to_fixed<PX, int32_t, 31>(PX *); \
template int32_t to_fixed<PX, int32_t, 32>(PX *); \
\
template int64_t to_fixed<PX, int64_t, 0>(PX *); \
template int64_t to_fixed<PX, int64_t, 1>(PX *); \
template int64_t to_fixed<PX, int64_t, 2>(PX *); \
template int64_t to_fixed<PX, int64_t, 3>(PX *); \
template int64_t to_fixed<PX, int64_t, 4>(PX *); \
template int64_t to_fixed<PX, int64_t, 5>(PX *); \
template int64_t to_fixed<PX, int64_t, 6>(PX *); \
template int64_t to_fixed<PX, int64_t, 7>(PX *); \
template int64_t to_fixed<PX, int64_t, 8>(PX *); \
template int64_t to_fixed<PX, int64_t, 9>(PX *); \
template int64_t to_fixed<PX, int64_t, 10>(PX *); \
template int64_t to_fixed<PX, int64_t, 11>(PX *); \
template int64_t to_fixed<PX, int64_t, 12>(PX *); \
template int64_t to_fixed<PX, int64_t, 13>(PX *); \
template int64_t to_fixed<PX, int64_t, 14>(PX *); \
template int64_t to_fixed<PX, int64_t, 15>(PX *); \
template int64_t to_fixed<PX, int64_t, 16>(PX *); \
template int64_t to_fixed<PX, int64_t, 17>(PX *); \
template int64_t to_fixed<PX, int64_t, 18>(PX *); \
template int64_t to_fixed<PX, int64_t, 19>(PX *); \
template int64_t to_fixed<PX, int64_t, 20>(PX *); \
template int64_t to_fixed<PX, int64_t, 21>(PX *); \
template int64_t to_fixed<PX, int64_t, 22>(PX *); \
template int64_t to_fixed<PX, int64_t, 23>(PX *); \
template int64_t to_fixed<PX, int64_t, 24>(PX *); \
template int64_t to_fixed<PX, int64_t, 25>(PX *); \
template int64_t to_fixed<PX, int64_t, 26>(PX *); \
template int64_t to_fixed<PX, int64_t, 27>(PX *); \
template int64_t to_fixed<PX, int64_t, 28>(PX *); \
template int64_t to_fixed<PX, int64_t, 29>(PX *); \
template int64_t to_fixed<PX, int64_t, 30>(PX *); \
template int64_t to_fixed<PX, int64_t, 31>(PX *); \
template int64_t to_fixed<PX, int64_t, 32>(PX *); \
template int64_t to_fixed<PX, int64_t, 33>(PX *); \
template int64_t to_fixed<PX, int64_t, 34>(PX *); \
template int64_t to_fixed<PX, int64_t, 35>(PX *); \
template int64_t to_fixed<PX, int64_t, 36>(PX *); \
template int64_t to_fixed<PX, int64_t, 37>(PX *); \
template int64_t to_fixed<PX, int64_t, 38>(PX *); \
template int64_t to_fixed<PX, int64_t, 39>(PX *); \
template int64_t to_fixed<PX, int64_t, 40>(PX *); \
template int64_t to_fixed<PX, int64_t, 41>(PX *); \
template int64_t to_fixed<PX, int64_t, 42>(PX *); \
template int64_t to_fixed<PX, int64_t, 43>(PX *); \
template int64_t to_fixed<PX, int64_t, 44>(PX *); \
template int64_t to_fixed<PX, int64_t, 45>(PX *); \
template int64_t to_fixed<PX, int64_t, 46>(PX *); \
template int64_t to_fixed<PX, int64_t, 47>(PX *); \
template int64_t to_fixed<PX, int64_t, 48>(PX *); \
template int64_t to_fixed<PX, int64_t, 49>(PX *); \
template int64_t to_fixed<PX, int64_t, 50>(PX *); \
template int64_t to_fixed<PX, int64_t, 51>(PX *); \
template int64_t to_fixed<PX, int64_t, 52>(PX *); \
template int64_t to_fixed<PX, int64_t, 53>(PX *); \
template int64_t to_fixed<PX, int64_t, 54>(PX *); \
template int64_t to_fixed<PX, int64_t, 55>(PX *); \
template int64_t to_fixed<PX, int64_t, 56>(PX *); \
template int64_t to_fixed<PX, int64_t, 57>(PX *); \
template int64_t to_fixed<PX, int64_t, 58>(PX *); \
template int64_t to_fixed<PX, int64_t, 59>(PX *); \
template int64_t to_fixed<PX, int64_t, 60>(PX *); \
template int64_t to_fixed<PX, int64_t, 61>(PX *); \
template int64_t to_fixed<PX, int64_t, 62>(PX *); \
template int64_t to_fixed<PX, int64_t, 63>(PX *); \
template int64_t to_fixed<PX, int64_t, 64>(PX *); \

EXPAND_FROM_FIXED(P8)
EXPAND_TO_FIXED(P8)
EXPAND_FROM_FIXED(P16)
EXPAND_TO_FIXED(P16)
EXPAND_FROM_FIXED(P32)
EXPAND_TO_FIXED(P32)

}
