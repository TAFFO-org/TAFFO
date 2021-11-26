//===-- AffineForms.cpp - Classes related to error propagation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of members of non-template classes
/// that perform error propagation with affine forms.
///
//===----------------------------------------------------------------------===//

#include "AffineForms.h"

ErrorProp::NoiseTermBase::SymbolT ErrorProp::NoiseTermBase::SymId = 0;
