#include "posit.h"

#define P32 Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf>
#define P16 Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf>
#define P8 Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf>

template class P32;
template class P16;
template class P8;

template P32::Posit(const P16 & other);
template P32::Posit(const P8 & other);
template P16::Posit(const P32 & other);
template P16::Posit(const P8 & other);
template P8::Posit(const P32 & other);
template P8::Posit(const P16 & other);

int main() {
    P16 x(3);
    P8 y = x;
}
