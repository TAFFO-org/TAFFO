#include "taffoAnnotations.hpp"

#include <fstream>
#include <iostream>

namespace taffo
{
	StructAnnotation parseStruct(const llvm::json::Array &array)
	{
		StructAnnotation toReturn;
		for (auto &elem : array)
		{
			auto subStructName = elem.getAsString();
			if (subStructName.hasValue())
			{
				toReturn.push_back(subStructName->str());
				continue;
			}

			auto primitiveAnnotations = elem.getAsObject();
			if (primitiveAnnotations != nullptr)
			{
				toReturn.push_back(Annotation(*primitiveAnnotations));
				continue;
			}
		}
		return toReturn;
	}

	void serialize(
			std::ostream &stream,
			const StructAnnotation &strct,
			const llvm::StringMap<StructAnnotation> &simbolTable)
	{
		stream << "struct[";
		for (auto val = strct.begin(); val != strct.end(); val++)
		{
			auto name = std::get_if<std::string>(&(*val));
			if (name != nullptr)
			{
				if (*name == "void")
					stream << "void";
				else
				{
					auto child = simbolTable.find(*name);
					if (child == simbolTable.end())
						continue;

					serialize(stream, child->getValue(), simbolTable);
				}
			}
			else
			{
				std::get_if<Annotation>(&(*val))->serialize(stream, &simbolTable);
			}
			if (val + 1 != strct.end())
				stream << ", ";
		}

		stream << "] ";
	}

	void serialize(
			std::ostream &stream,
			const StringOrAnnotation &ann,
			const llvm::StringMap<StructAnnotation> &simbolTable)
	{
		auto a = std::get_if<Annotation>(&ann);
		if (a != nullptr)
			a->serialize(stream, &simbolTable);
		else
		{
			std::string name = *std::get_if<std::string>(&ann);
			auto child = simbolTable.find(name);
			if (child == simbolTable.end())
			{
				llvm::errs() << "struct " << name << " not found\n";
				return;
			}
			serialize(stream, child->second, simbolTable);
		}
	}

	void toJSON(std::ostream &stream, const StructAnnotation &strct)
	{
		stream << "\"content\": [{}, ";
		for (auto val = strct.begin(); val != strct.end(); val++)
		{
			auto name = std::get_if<std::string>(&(*val));
			if (name != nullptr)
				stream << "\"" << *name << "\"";
			else
			{
				stream << "{\n";
				std::get_if<Annotation>(&(*val))->toJSON(stream);
				stream << "}\n";
			}

			if (val + 1 != strct.end())
				stream << ", ";
		}
		stream << "]";
	}
	void toJSON(std::ostream &stream, const StringOrAnnotation &ann)
	{
		auto a = std::get_if<Annotation>(&ann);
		if (a != nullptr)
		{
			a->toJSON(stream);
		}
		else
		{
			stream << "\"struct\": \"";
			stream << *std::get_if<std::string>(&ann);
			stream << "\"\n";
		}
	}
	AnnotationMap::AnnotationMap(const llvm::json::Array &root)
	{
		for (unsigned a = 0; a < root.size(); a++)
		{
			auto child = root[a].getAsObject();

			if (child == nullptr)
			{
				llvm::errs() << "Top level item of json was not an object\n";
				continue;
			}
			if (child->empty())
				continue;

			auto structName = child->getString("struct");
			auto localVarName = child->getString("localVar");
			auto globalVarName = child->getString("globalVar");
			auto functionName = child->getString("function");

			if (localVarName.hasValue())
			{
				if (!functionName.hasValue())
				{
					llvm::errs() << "function name for local var is missing\n";
					continue;
				}
				localAnnotations[functionName.getValue()][localVarName.getValue()] =
						Annotation(*child);
				continue;
			}
			if (globalVarName.hasValue())
			{
				globalAnnotations[globalVarName.getValue()] = Annotation(*child);
				continue;
			}

			if (functionName.hasValue())
			{
				functionAnnotations[functionName.getValue()] = Annotation(*child);
				continue;
			}

			if (structName.hasValue())
			{
				auto structArray = child->getArray("content");
				if (structArray != nullptr)
					symbolTable[structName.getValue()] = parseStruct(*structArray);
				else
					llvm::errs() << "struct was declared without content\n";
				continue;
			}
		}
	}
	void AnnotationMap::toJSON(std::ostream &stream)
	{
		stream << "[\n";

		for (auto const &x : symbolTable)
		{
			stream << "{\n";
			stream << "\"struct\": \"" << x.first().str() << "\",\n";
			taffo::toJSON(stream, x.second);
			stream << "},\n";
		}

		for (auto const &x : functionAnnotations)
		{
			stream << "{\n";
			stream << "\"function\": \"" << x.first().str() << "\",\n";
			taffo::toJSON(stream, x.second);
			stream << "},\n";
		}

		for (auto const &x : globalAnnotations)
		{
			stream << "{\n";
			stream << "\"globalVar\": \"" << x.first().str() << "\",\n";
			taffo::toJSON(stream, x.second);
			stream << "},\n";
		}

		for (auto const &x : localAnnotations)
		{
			for (auto const &f : x.second)
			{
				stream << "{\n";
				stream << "\"localVar\": \"" << x.first().str() << "\",\n";
				stream << "\"function\": \"" << f.first().str() << "\",\n";
				taffo::toJSON(stream, f.second);
				stream << "},\n";
			}
		}

		stream << "{}\n";
		stream << "]\n";
	}
};	// namespace taffo
