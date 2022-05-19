/*
 * tritri.hpp
 * 
 * Created on: Sep 9, 2013
 * 			Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */

#ifndef __TRITRI_HPP__
#define __TRITRI_HPP__
#include <cstdlib>
#include <iostream>

//#define INSTRUMENT
#ifdef INSTRUMENT
#define PRINT_INSTR(...) fprintf(stderr, __VA_ARGS__)
#else
#define PRINT_INSTR(...) do {} while (false);
#endif

int tri_tri_intersect(float V0[3], float V1[3], float V2[3],
		      float U0[3], float U1[3], float U2[3],
		      float *res, int *output) ;

#endif
