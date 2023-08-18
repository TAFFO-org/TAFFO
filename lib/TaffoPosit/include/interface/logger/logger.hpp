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
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
typedef enum LogLevel {
	ALL,STAT,NONE
} LogLevel;

template <class T>
class Logger
{
private:
	std::vector<T> _valueStorage;
	T _maxVal;
	T _minVal;
	T _meanVal;
	T _count=0;
	LogLevel _level = LogLevel::NONE;
public:
	void logValue(T val) {
		if(_level == LogLevel::ALL || _level == LogLevel::STAT) {
			if( _count==0 )
				_maxVal = _minVal = _meanVal = val;
			else {
				_maxVal = (val > _maxVal)? val:_maxVal;
				_minVal = (val < _minVal)? val:_minVal;
				_meanVal = (_meanVal*_count+val)/(_count+1);
			}
			_count++;
		}
		if(_level == LogLevel::ALL) 
			_valueStorage.push_back(val);

	}	

	void setLogLevel(LogLevel l) {
		_level = l;
	}

	void save(const std::string& path) {
		std::ofstream of(path);
		of << "===========================================" << std::endl;
		of << "Max: " << _maxVal << " Min: " << _minVal << " Mean: " << _meanVal << std::endl;
		of << "===========================================" << std::endl;		
		for(T val:_valueStorage)
			of << val << std::endl;
		of.flush();
		of.close();
	}
};