#pragma once

//Precision to use for calculations
#ifdef EMBEDDED
#define fptype float
#else
#define fptype double
#endif

struct blackscholes_line {
  fptype s, stk, r, divq, v, t;
  char OptionType;
  fptype divs, DGrefval;
};

#define BLACKSCHOLES_DATA_SIZE 2000
extern const blackscholes_line blackscholes_data[BLACKSCHOLES_DATA_SIZE];

