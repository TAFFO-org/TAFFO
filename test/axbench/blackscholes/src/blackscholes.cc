// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
//
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice
// Hall, John C. Hull,

#include "benchmark.hpp"
#include "data.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


// double max_otype, min_otype ;
// double max_sptprice, min_sptprice;
// double max_strike, min_strike;
// double max_rate, min_rate ;
// double max_volatility, min_volatility;
// double max_otime, min_otime ;
// double max_out_price, min_out_price;

#define DIVIDE 120.0

#define NUM_RUNS 1

static fptype *prices;
static int numOptions;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286f

static fptype CNDF(fptype InputX)
{
  int sign;

  fptype __attribute((annotate("scalar()"))) OutputX;
  fptype __attribute((annotate("scalar()"))) xInput;
  fptype __attribute((annotate("scalar()"))) xNPrimeofX;
  fptype __attribute((annotate("scalar(range(0,1) final)"))) expValues;
  fptype __attribute((annotate("scalar()"))) xK2;
  fptype __attribute((annotate("scalar()"))) xK2_2, xK2_3;
  fptype __attribute((annotate("scalar()"))) xK2_4, xK2_5;
  fptype __attribute((annotate("scalar()"))) xLocal, xLocal_1;
  fptype __attribute((annotate("scalar()"))) xLocal_2, xLocal_3;
  fptype __attribute((annotate("scalar(range(30,12412) final)"))) InputXX;

  // Check for negative value of InputX
  if (InputX < 0.0) {
    InputX = -InputX;
    sign = 1;
  } else
    sign = 0;

  xInput = InputX;
  InputXX = InputX;

  // Compute NPrimeX term common to both four & six decimal accuracy calcs
  expValues = exp(-0.5f * (InputXX * InputXX));
  xNPrimeofX = expValues;
  xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

  xK2 = 0.2316419f * InputXX;
  xK2 = 1.0f + xK2;
  xK2 = 1.0f / xK2;
  xK2_2 = xK2 * xK2;
  xK2_3 = xK2_2 * xK2;
  xK2_4 = xK2_3 * xK2;
  xK2_5 = xK2_4 * xK2;

  xLocal_1 = xK2 * 0.319381530f;
  xLocal_2 = xK2_2 * (-0.356563782f);
  xLocal_3 = xK2_3 * 1.781477937f;
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_4 * (-1.821255978f);
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_5 * 1.330274429f;
  xLocal_2 = xLocal_2 + xLocal_3;

  xLocal_1 = xLocal_2 + xLocal_1;
  xLocal = xLocal_1 * xNPrimeofX;

  // printf("# xLocal: %10.10f\n", xLocal);

  xLocal = 1.0f - xLocal;

  OutputX = xLocal;

  // printf("# Output: %10.10f\n", OutputX);

  if (sign) {
    OutputX = 1.0f - OutputX;
  }

  return OutputX;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
static fptype BlkSchlsEqEuroNoDiv(fptype  sptprice,
                                  fptype  strike,
                                  fptype  rate,
                                  fptype  volatility,
                                  fptype  time,
                                  int otype)
{
  // printf("BlkSchlsEqEuroNoDiv %f %f %f %f %f %f\n", sptprice, strike, rate, volatility, time, timet);
  fptype __attribute((annotate("scalar()"))) OptionPrice;

  // local private working variables for the calculation
  // fptype xStockPrice;
  // fptype xStrikePrice;
  fptype __attribute((annotate("scalar( range(0.05, 0.1))"))) xRiskFreeRate;
  fptype __attribute((annotate("scalar( range(0.1, 0.5))"))) xVolatility;
  fptype __attribute((annotate("scalar( range(0.1, 1))"))) xTime;
  fptype __attribute((annotate("scalar( range(0.30, 1))"))) xSqrtTime;

  fptype __attribute((annotate("scalar(range(-0.20, 0.20))"))) logValues;
  fptype __attribute((annotate("scalar()"))) xLogTerm;
  fptype __attribute((annotate("scalar()"))) xD1;
  fptype __attribute((annotate("scalar()"))) xD2;
  fptype __attribute((annotate("scalar()"))) xPowerTerm;
  fptype __attribute((annotate("scalar()"))) xDen;
  fptype __attribute((annotate("scalar(range(-5.9 6.5))"))) d1;
  fptype __attribute((annotate("scalar()"))) d2;
  fptype __attribute((annotate("scalar()"))) FutureValueX;
  fptype __attribute((annotate("scalar()"))) NofXd1;
  fptype __attribute((annotate("scalar()"))) NofXd2;
  fptype __attribute((annotate("scalar()"))) NegNofXd1;
  fptype __attribute((annotate("scalar()"))) NegNofXd2;
  fptype __attribute((annotate("scalar()"))) division;


  xRiskFreeRate = rate;
  xVolatility = volatility;
  xTime = time;


  xSqrtTime = sqrt(xTime);
  
  division = sptprice / strike;
  logValues = log(division);

  


  xPowerTerm = xVolatility * xVolatility;
  xPowerTerm = xPowerTerm * 0.5f;

  xD1 = xRiskFreeRate + xPowerTerm;
  xD1 = xD1 * xTime;
  xD1 = xD1 + logValues;

  xDen = xVolatility * xSqrtTime;
  xD1 = xD1 / xDen;
  xD2 = xD1 - xDen;

  d1 = xD1;
  d2 = xD2;


  NofXd1 = CNDF(d1);



  if (NofXd1 > 1.0) {
    // std::cerr << "Greater than one!" << std::endl ;
  }
  // printf("# d1: %10.10f\n", NofXd1);

  NofXd2 = CNDF(d2);
  if (NofXd2 > 1.0) {
    // std::cerr << "Greater than one!" << std::endl ;
  }
  // printf("# d2: %10.10f\n", NofXd2);


  fptype __attribute((annotate("scalar( range(-1, 1) )"))) e =  -(rate) * (time);
  fptype __attribute((annotate("scalar(  )"))) e2 =  (1+e+e*e/2);

  FutureValueX = strike * e2;

  if (otype == 0) {
    fptype __attribute((annotate("scalar(0 1)"))) t1 = (sptprice * NofXd1);
    fptype __attribute((annotate("scalar(0 1)"))) t2 = (FutureValueX * NofXd2);
    OptionPrice = t1 - t2 ;


  } else {
    NegNofXd1 = (1.0f - NofXd1);
    NegNofXd2 = (1.0f - NofXd2);
    fptype __attribute((annotate("scalar(0 1)"))) t1 = (FutureValueX * NegNofXd2);
    fptype __attribute((annotate("scalar(0 1)"))) t2 = (sptprice * NegNofXd1);
    OptionPrice = t1- t2;

  }

  return OptionPrice;
}


static fptype normalize(fptype in, fptype min, fptype max, fptype min_new, fptype max_new)
{
  return (((in - min) / (max - min)) * (max_new - min_new)) + min_new;
}

static int bs_thread(fptype *sptprice, fptype *strike, fptype *rate, fptype *volatility, fptype *otime, int *otype)
{
  int i, j;


  fptype __attribute((annotate("scalar(range(-1, 1))"))) price_orig;

  for (j = 0; j < NUM_RUNS; j++) {
    for (i = 0; i < numOptions; i++) {
      /* Calling main function to calculate option value based on
       * Black & Scholes's equation.
       */

      price_orig = BlkSchlsEqEuroNoDiv(sptprice[i], strike[i],
                                       rate[i], volatility[i], otime[i],
                                       otype[i]);

      prices[i] = price_orig;
    }
  }
  return 0;
}

int main(int argc, char **argv)
{
  int i;
  int loopnum;
  fptype *buffer;
  int *buffer2;
  int rv;
  rv = numOptions = BLACKSCHOLES_DATA_SIZE;

  int *otype;
  fptype __attribute((annotate("target('a') scalar(range(0,1) )"))) * sptprice;
  fptype __attribute((annotate("scalar(range(0,1) )"))) * strike;
  fptype __attribute((annotate("scalar(range(0.0275,0.1) )"))) * rate;
  fptype __attribute((annotate("scalar(range(0.05,0.65) )"))) * volatility;
  fptype __attribute((annotate("scalar(range(0.05,1) )"))) * otime;
  // alloc spaces for the option data
  prices = (fptype *)malloc(numOptions * sizeof(fptype));

#define PAD 256
#define LINESIZE 64

  sptprice = (fptype *)malloc( numOptions * sizeof(fptype));
  strike = (fptype *)malloc( numOptions * sizeof(fptype));
  rate = (fptype *)malloc( numOptions * sizeof(fptype));
  volatility = (fptype *)malloc( numOptions * sizeof(fptype));
  otime = (fptype *)malloc( numOptions * sizeof(fptype));

  otype = (int *)malloc(numOptions * sizeof(fptype) );

  for (i = 0; i < numOptions; i++) {
    otype[i] = (blackscholes_data[i].OptionType == 'P') ? 1 : 0;
    sptprice[i] = blackscholes_data[i].s / DIVIDE;
    strike[i] = blackscholes_data[i].stk / DIVIDE;
    rate[i] = blackscholes_data[i].r;
    volatility[i] = blackscholes_data[i].v;
    otime[i] = blackscholes_data[i].t;
    //printf("sptprice: %f strike: %f rate: %f volatility: %f otime: %f \n", sptprice[i], strike[i], rate[i], volatility[i], otime[i]);
  }


  // serial version

  // AxBenchTimer timer;

  int tid = 0;
  bs_thread(sptprice, strike, rate, volatility, otime, otype);

  // uint64_t time = timer.nanosecondsSinceInit();
  // std::cout << "kernel time = " << time << " cycles\n";

  // Write prices to output file
  for (i = 0; i < numOptions; i++) {
     rv = printf("%.18f\n", prices[i]);
  }



  return 0;
}
