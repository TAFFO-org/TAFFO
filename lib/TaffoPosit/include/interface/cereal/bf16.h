/**
 * Copyright (C) 2017-2023 Emanuele Ruffaldi, Federico Rossi
 * 
 * Distributed under the terms of the BSD 3-Clause License.  
 * 
 * (See accompanying file LICENSE)
 * 
 * --
 */

#pragma once
#include <floats/anyfloat.hpp>

template<class Archive>
void save(Archive & archive, 
          binary16alt_emu const & p)
{ 
  archive((float)p); 
}

template<class Archive>
void load(Archive & archive,
          binary16alt_emu & p)
{
  float d;
  archive(d); 
  p = binary16alt_emu(d);
} 


template<class Archive>
void save(Archive & archive, 
          binary8_emu const & p)
{ 
  archive((float)p); 
}

template<class Archive>
void load(Archive & archive,
          binary8_emu & p)
{
  float d;
  archive(d); 
  p = binary8_emu(d);
} 