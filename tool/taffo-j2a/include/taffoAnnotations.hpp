#pragma once
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>
#include <map>
#include <sstream>
#include <vector>

#include "annotation.hpp"

namespace taffo
{
	StructAnnotation parseStruct(const llvm::json::Array &array);
	void serialize(
			std::ostream &stream,
			const StructAnnotation &strct,
			const llvm::StringMap<StructAnnotation> &simbolTable);
	void serialize(
			std::ostream &stream,
			const StringOrAnnotation &ann,
			const llvm::StringMap<StructAnnotation> &simbolTable);
	void toJSON(std::ostream &stream, const StructAnnotation &strct);
	void toJSON(std::ostream &stream, const StringOrAnnotation &ann);

	class AnnotationMap
	{
		public:
		AnnotationMap(const llvm::json::Array &root);
		AnnotationMap() = default;

		std::string globalToStr(llvm::StringRef name)
		{
			std::stringstream ss;
			if (globalExists(name))
			{
				serialize(ss, globalAnnotations[name], symbolTable);
				return ss.str();
			}
			return "";
		}

		std::string localToStr(llvm::StringRef name, llvm::StringRef function)
		{
			std::stringstream ss;
			if (localExists(name, function))
			{
				serialize(ss, localAnnotations[function][name], symbolTable);
				return ss.str();
			}
			return "";
		}

		std::string functionToStr(llvm::StringRef name)
		{
			std::stringstream ss;
			if (functionExists(name))
			{
				serialize(ss, functionAnnotations[name], symbolTable);
				return ss.str();
			}
			return "";
		}

		bool functionExists(llvm::StringRef name) const
		{
			return functionAnnotations.find(name) != functionAnnotations.end();
		}
		bool globalExists(llvm::StringRef name) const
		{
			return globalAnnotations.find(name) != globalAnnotations.end();
		}
		bool localExists(llvm::StringRef name, llvm::StringRef function) const
		{
			auto fun = localAnnotations.find(function);
			if (fun == localAnnotations.end())
				return false;
			return fun->getValue().find(name) != fun->getValue().end();
		}

		void insertGlobal(StringOrAnnotation annotation, llvm::StringRef varName)
		{
			globalAnnotations[varName] = std::move(annotation);
		}
		void insertFunction(
				StringOrAnnotation annotation, llvm::StringRef functionName)
		{
			functionAnnotations[functionName] = std::move(annotation);
		}
		void insertLocal(
				StringOrAnnotation annotation,
				llvm::StringRef varName,
				llvm::StringRef functionName)
		{
			localAnnotations[functionName][varName] = std::move(annotation);
		}

		void insertStruct(StructAnnotation strct, llvm::StringRef strName)
		{
			symbolTable[strName] = std::move(strct);
		}

		void toJSON(std::ostream &stream);

		private:
		llvm::StringMap<llvm::StringMap<StringOrAnnotation>> localAnnotations;
		llvm::StringMap<StringOrAnnotation> globalAnnotations;
		llvm::StringMap<StringOrAnnotation> functionAnnotations;
		llvm::StringMap<StructAnnotation> symbolTable;
	};
}	// namespace taffo
