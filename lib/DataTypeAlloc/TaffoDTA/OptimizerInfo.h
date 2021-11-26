#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "InputInfo.h"
#include "Metadata.h"
#include "TypeUtils.h"
#include "Infos.h"

#ifndef __TAFFO_DTA_OPTIMIZERINFO_H__
#define __TAFFO_DTA_OPTIMIZERINFO_H__

namespace tuner {
    using namespace std;
    class OptimizerInfo {
    public:
        enum OptimizerInfoKind {
            K_Struct, K_Field, K_Pointer
        };

        OptimizerInfo(OptimizerInfoKind K) : Kind(K) {}


        //virtual OptimizerInfo *clone() const = 0;

        virtual ~OptimizerInfo() = default;

        OptimizerInfoKind getKind() const { return Kind; }

        virtual std::string toString() const {
            return "OptimizerInfo";
        };

        virtual bool operator==(const OptimizerInfo &b) const {
            return Kind == b.Kind;
        }


    private:
        const OptimizerInfoKind Kind;
    };

/// Structure containing pointers to Type, Range, and initial Error
/// of an LLVM Value.
    struct OptimizerScalarInfo : public OptimizerInfo {
        std::shared_ptr<string> baseName;
        unsigned minBits;
        unsigned maxBits;
        unsigned totalBits;
        bool isSigned;
        string overridedEnob;
        shared_ptr<mdutils::Range> range;

        bool referToConstant;


        OptimizerScalarInfo(string _variableName, unsigned _minBits, unsigned _maxBits, unsigned _totalBits,
                            bool _isSigned, mdutils::Range _range, string _overriddenEnob)
                : OptimizerInfo(K_Field), referToConstant(false) {
            minBits = _minBits;
            maxBits = _maxBits;
            baseName = make_shared<string>(_variableName);
            totalBits = _totalBits;
            isSigned = _isSigned;
            range = make_shared<mdutils::Range>(_range);
            overridedEnob = _overriddenEnob;
        }

        shared_ptr<mdutils::Range> getRange() {
            return range;
        }

        const string getBaseName() const {
            return *baseName.get();
        }

        string getOverridedEnob() {
            return overridedEnob;
        }

        bool doesReferToConstant() const {
            return referToConstant;
        }

        void setReferToConstant(bool referToConstant) {
            this->referToConstant = referToConstant;
        }


        virtual std::string toString() const override {
            std::stringstream sstm;
            sstm << "ScalarInfo(";
            sstm << *(baseName.get());
            sstm << ", ";
            sstm << overridedEnob;
            sstm << ")";
            return sstm.str();
        };

        unsigned int getMinBits() const {
            return minBits;
        }

        unsigned int getMaxBits() const {
            return maxBits;
        }

        const string getFixedSelectedVariable() {
            return *baseName + "_fixp";
        }

        const string getFloatSelectedVariable() {
            return *baseName + "_float";
        }

        const string getDoubleSelectedVariable() {
            return *baseName + "_double";
        }

        const string getHalfSelectedVariable() {
            return *baseName + "_Half";
        }

        const string getQuadSelectedVariable() {
            return *baseName + "_Quad";
        }

        const string getFP80SelectedVariable() {
            return *baseName + "_FP80";
        }

        const string getPPC128SelectedVariable() {
            return *baseName + "_PPC128";
        }

        const string getBF16SelectedVariable() {
            return *baseName + "_BF16";
        }        

        const string getFractBitsVariable() {
            return *baseName + "_fixbits";
        }

        const string getRealEnobVariable() {
            if (!overridedEnob.empty()) {
                return overridedEnob;
            }
            return *baseName + "_enob";
        }

        const string getBaseEnobVariable() {
            return *baseName + "_enob";
        }

        unsigned int getTotalBits() const {
            return totalBits;
        }

        bool isSigned1() const {
            return isSigned;
        }

        void overrideEnob(string newEnob) {
            overridedEnob = newEnob;
        }

        bool operator==(const OptimizerInfo &other) const override {
            if (!OptimizerInfo::operator==(other)) {
                return false;
            }

            auto *b2 = llvm::cast<OptimizerScalarInfo>(&other);
            return baseName == b2->baseName;
        }


        static bool classof(const OptimizerInfo *M) { return M->getKind() == K_Field; }
    };

    class OptimizerStructInfo : public OptimizerInfo {
    private:
        typedef llvm::SmallVector<std::shared_ptr<OptimizerInfo>, 4U> FieldsType;
        FieldsType Fields;


    public:
        typedef FieldsType::iterator iterator;
        typedef FieldsType::const_iterator const_iterator;
        typedef FieldsType::size_type size_type;

        OptimizerStructInfo(int size)
                : OptimizerInfo(K_Struct), Fields(size, nullptr) {}

        OptimizerStructInfo(const llvm::ArrayRef<std::shared_ptr<OptimizerInfo>> SInfos)
                : OptimizerInfo(K_Struct), Fields(SInfos.begin(), SInfos.end()) {}

        iterator begin() { return Fields.begin(); }

        iterator end() { return Fields.end(); }

        const_iterator begin() const { return Fields.begin(); }

        const_iterator end() const { return Fields.end(); }

        size_type size() const { return Fields.size(); }

        OptimizerInfo *getField(size_type I) const { return Fields[I].get(); }

        void setField(size_type I, std::shared_ptr<OptimizerInfo> F) { Fields[I] = F; }

        std::shared_ptr<OptimizerInfo> getField(size_type I) { return Fields[I]; }

        std::shared_ptr<OptimizerInfo> resolveFromIndexList(llvm::Type *type, llvm::ArrayRef<unsigned> indices) {
            llvm::Type *resolvedType = type;
            std::shared_ptr<OptimizerInfo> resolvedInfo(this);
            for (unsigned idx: indices) {
                if (resolvedInfo.get() == nullptr)
                    break;
                if (resolvedType->isStructTy()) {
                    resolvedType = resolvedType->getContainedType(idx);
                    resolvedInfo = llvm::cast<OptimizerStructInfo>(resolvedInfo.get())->getField(idx);
                } else {
                    resolvedType = resolvedType->getContainedType(idx);
                }
            }
            return resolvedInfo;
        }


        virtual std::string toString() const override {
            std::stringstream sstm;
            sstm << "StructInfo(";
            int i = 0;
            for (auto inf : Fields) {
                if (i) {
                    sstm << "; ";
                }
                if (inf) {
                    sstm << inf->toString();
                } else {
                    sstm << "nullptr";
                }
                i++;
            }
            sstm << ")";
            return sstm.str();
        };

        bool operator==(const OptimizerInfo &other) const override {
            if (!OptimizerInfo::operator==(other)) {
                return false;
            }

            auto *b2 = llvm::cast<OptimizerStructInfo>(&other);
            if (Fields.size() != b2->Fields.size()) {
                return false;
            }

            for (unsigned int i = 0; i < Fields.size(); i++) {
                if (Fields[i] == nullptr && b2->Fields[i] == nullptr) {
                    continue;
                }
                if (Fields[i] == nullptr) {
                    return false;
                }
                if (b2->Fields[i] == nullptr) {
                    return false;
                }
                if (!Fields[i]->operator==(*b2->Fields[i])) {
                    return false;
                }

            }

            return true;
        }


        static bool classof(const OptimizerInfo *M) { return M->getKind() == K_Struct; }
    };


/// Structure containing pointers to Type, Range, and initial Error
/// of an LLVM Value.
    struct OptimizerPointerInfo : public OptimizerInfo {
    private:
        std::shared_ptr<OptimizerInfo> optInfo;
    public:
        OptimizerPointerInfo(shared_ptr<OptimizerInfo> pointee)
                : OptimizerInfo(K_Pointer) {
            assert(pointee && "Pointee should not be null!");
            optInfo = pointee;
        }

        const shared_ptr<OptimizerInfo> &getOptInfo() const {
            return optInfo;
        }


        virtual std::string toString() const override {
            std::stringstream sstm;
            sstm << "PointerInfo(";
            sstm << optInfo->toString();
            sstm << ")";
            return sstm.str();
        };


        static bool classof(const OptimizerInfo *M) { return M->getKind() == K_Pointer; }
    };

}

#endif
