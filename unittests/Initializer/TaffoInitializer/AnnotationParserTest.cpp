#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "TaffoInitializer/AnnotationParser.h"


namespace
{
using namespace taffo;
using namespace llvm;

class AnnotationParserTest : public testing::Test
{
protected:
  AnnotationParser parser;
  llvm::StringRef annstr;
  std::string error;
  bool res;
  mdutils::InputInfo *scalarMD;
  mdutils::StructInfo *structMD;

  // error messages
  const std::string NO_DTP_ERR = "scalar() or struct() top-level specifiers missing";
  const std::string DUP_DTP_ERR = "Duplicated content definition in this context";
  const std::string UNK_ID_ERR = "Unknown identifier at character index ";
  const std::string NO_STR_ERR = "Expected string at character index ";
  const std::string NO_INT_ERR = "Expected integer at character index ";
  const std::string NO_REAL_ERR = "Expected real at character index ";
  const std::string NO_BOOL_ERR = "Expected boolean at character index ";
  const std::string OLD_SYN_ERR = "Somebody used the old syntax and they should stop.";
  const std::string EMPTY_STRUCT_ERR = "Empty structures not allowed";
  const std::string NO_ERR = "";
};


/**
 * Valid syntax for target:
 * - target(<string>) <DTP>
 */
TEST_F(AnnotationParserTest, TargetScalar)
{
  // valid, with a scalar DTP
  annstr = llvm::StringRef("target('this is a valid case') scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);


  EXPECT_EQ("this is a valid case", parser.target.getValue());
  EXPECT_TRUE(parser.startingPoint);
}

TEST_F(AnnotationParserTest, TargetStruct)
{
  // valid, with a struct DTP
  annstr = llvm::StringRef("target('this is a valid case') struct[void]");
  res = parser.parseAnnotationString(annstr);
  ASSERT_TRUE(res);
  error = parser.lastError().str();

  EXPECT_EQ(NO_ERR, error);
  EXPECT_EQ("this is a valid case", parser.target.getValue());
  EXPECT_TRUE(parser.startingPoint);
}

TEST_F(AnnotationParserTest, TargetNone)
{
  // not valid, no DTP
  annstr = llvm::StringRef("target('this is not valid')");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, TargetVoid)
{
  // not valid, void not valid at top level (unknown identifier)
  annstr = llvm::StringRef("target('this is not valid') void");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_THAT(error, testing::StartsWith(UNK_ID_ERR));
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, TargetNoString)
{
  // not valid, no target string
  annstr = llvm::StringRef("target() scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_THAT(error, testing::StartsWith(NO_STR_ERR));
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, TargetRepeated)
{
  // not valid, repeated target twice
  annstr = llvm::StringRef("target('this is not valid') target('not valid)' scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_THAT(error, testing::StartsWith(UNK_ID_ERR));
  ASSERT_FALSE(res);
}


/**
 * Valid syntax for errtarget:
 * - errtarget(<string>) <DTP>
 */
TEST_F(AnnotationParserTest, ErrTargetScalar)
{
  // valid, with a scalar DTP
  annstr = llvm::StringRef("errtarget('this is a valid case') scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ("this is a valid case", parser.target.getValue());
  EXPECT_FALSE(parser.startingPoint);
}

TEST_F(AnnotationParserTest, ErrTargetStruct)
{
  // valid, with a struct DTP
  annstr = llvm::StringRef("errtarget('this is a valid case') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_FALSE(parser.startingPoint);
}

TEST_F(AnnotationParserTest, ErrTargetNone)
{
  // not valid, no DTP
  annstr = llvm::StringRef("errtarget('this is not valid')");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_THAT(error, testing::StartsWith(NO_DTP_ERR));
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, ErrTargetVoid)
{
  // not valid, void not valid at top level (unknown identifier)
  annstr = llvm::StringRef("errtarget('this is not valid') void");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_THAT(error, testing::StartsWith(UNK_ID_ERR));
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, ErrTargetNoString)
{
  // not valid, no target string
  annstr = llvm::StringRef("errtarget() scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
  EXPECT_THAT(error, testing::StartsWith(NO_STR_ERR));
}

TEST_F(AnnotationParserTest, ErrTargetRepeated)
{
  // not valid, repeated errtarget twice
  annstr = llvm::StringRef("errtarget('this is not valid') errtarget('not valid)' scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}


/**
 * Valid syntax for backtracking:
 * - backtracking <DTP>
 * - backtracking(<int>) <DTP>
 * - backtracking(<bool>) <DTP>
 */
TEST_F(AnnotationParserTest, BacktrackingScalarNoParam)
{
  // valid, with a scalar DTP
  annstr = llvm::StringRef("backtracking scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_TRUE(parser.backtracking);
  EXPECT_EQ(std::numeric_limits<unsigned int>::max(), parser.backtrackingDepth);
}

TEST_F(AnnotationParserTest, BacktrackingStructNoParam)
{
  // valid, with a struct DTP
  annstr = llvm::StringRef("backtracking struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_TRUE(parser.backtracking);
  EXPECT_EQ(std::numeric_limits<unsigned int>::max(), parser.backtrackingDepth);
}

TEST_F(AnnotationParserTest, BacktrackingInteger)
{
  // valid, with a scalar DTP
  annstr = llvm::StringRef("backtracking(12) scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_TRUE(parser.backtracking);
  EXPECT_EQ(12, parser.backtrackingDepth);
}

TEST_F(AnnotationParserTest, BacktrackingIntegerNegative)
{
  // valid, with a scalar DTP
  annstr = llvm::StringRef("backtracking(-12) scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_TRUE(parser.backtracking);
  EXPECT_EQ(-12, parser.backtrackingDepth);
}

TEST_F(AnnotationParserTest, BacktrackingBoolFalse)
{
  // valid, with a scalar DTP
  annstr = llvm::StringRef("backtracking(false) scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_FALSE(parser.backtracking);
  // backtrackingDepth is not set
}

TEST_F(AnnotationParserTest, BacktrackingBoolTrue)
{
  // valid, with a scalar DTP
  annstr = llvm::StringRef("backtracking(true) scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_TRUE(parser.backtracking);
  // backtrackingDepth is not set
}

TEST_F(AnnotationParserTest, BacktrackingZero)
{
  // valid, with a struct DTP
  annstr = llvm::StringRef("backtracking(0) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_FALSE(parser.backtracking);
  // backtrackingDepth is not set
}

TEST_F(AnnotationParserTest, BacktrackingNone)
{
  // not valid, no DTP
  annstr = llvm::StringRef("backtracking");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  // EXPECT_THAT(error, testing::StartsWith(NO_DTP_ERR));
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, BacktrackingVoid)
{
  // not valid, void not valid at top level (unknown identifier)
  annstr = llvm::StringRef("backtracking void");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  // EXPECT_THAT(error, testing::StartsWith(UNK_ID_ERR));
  ASSERT_FALSE(res);
}


/**
 * Valid syntax for a scalar():
 * - scalar()
 * - scalar(<data_attr>)
 */
TEST_F(AnnotationParserTest, ScalarNone)
{
  // valid, no data_attr
  annstr = llvm::StringRef("scalar()");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(nullptr, scalarMD->IType);
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarRange)
{
  // valid, with range()
  annstr = llvm::StringRef("scalar(range(0, 10))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(nullptr, scalarMD->IType);
  EXPECT_EQ(0, scalarMD->IRange->Min);
  EXPECT_EQ(10, scalarMD->IRange->Max);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarType)
{
  // valid, with type()
  annstr = llvm::StringRef("scalar(type(1 2))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(1, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getWidth());
  EXPECT_EQ(2, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getPointPos());
  EXPECT_EQ(true, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->isSigned());
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarTypeSigned)
{
  // valid, with type()
  annstr = llvm::StringRef("scalar(type(signed 1 2))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(1, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getWidth());
  EXPECT_EQ(2, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getPointPos());
  EXPECT_EQ(true, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->isSigned());
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarTypeSignedNegative)
{
  // valid, with type()
  annstr = llvm::StringRef("scalar(type(signed -1 2))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(1, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getWidth());
  EXPECT_EQ(2, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getPointPos());
  EXPECT_EQ(true, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->isSigned());
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarTypeUnsigned)
{
  // valid, with type()
  annstr = llvm::StringRef("scalar(type(unsigned 1 2))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(1, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getWidth());
  EXPECT_EQ(2, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getPointPos());
  EXPECT_EQ(false, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->isSigned());
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarTypeUnsignedNegative)
{
  // valid, with type()
  annstr = llvm::StringRef("scalar(type(unsigned -1 2))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(1, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getWidth());
  EXPECT_EQ(2, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getPointPos());
  EXPECT_EQ(false, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->isSigned());
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarError)
{
  // valid, with error()
  annstr = llvm::StringRef("scalar(error(3.1415))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(nullptr, scalarMD->IType);
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(3.1415, *scalarMD->IError.get());
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarDisabled)
{
  // valid, with disabled
  annstr = llvm::StringRef("scalar(disabled)");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(nullptr, scalarMD->IType);
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(false, scalarMD->IEnableConversion);
  EXPECT_EQ(false, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarFinal)
{
  // valid, with final
  annstr = llvm::StringRef("scalar(final)");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(nullptr, scalarMD->IType);
  EXPECT_EQ(nullptr, scalarMD->IRange);
  EXPECT_EQ(nullptr, scalarMD->IError);
  EXPECT_EQ(true, scalarMD->IEnableConversion);
  EXPECT_EQ(true, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarMultipleDTA)
{
  // valid, multiple data attributes
  annstr = llvm::StringRef("scalar(range(0, 10) type(1 2) error(3.1415) disabled final)");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(1, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getWidth());
  EXPECT_EQ(2, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getPointPos());
  EXPECT_EQ(true, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->isSigned());
  EXPECT_EQ(0, scalarMD->IRange->Min);
  EXPECT_EQ(10, scalarMD->IRange->Max);
  EXPECT_EQ(3.1415, *scalarMD->IError.get());
  EXPECT_EQ(false, scalarMD->IEnableConversion);
  EXPECT_EQ(true, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarRepeatedAttributes)
{
  // valid, but the metadata considers only the last encountered scalar()
  annstr = llvm::StringRef("scalar(range(0, 10) type(1 2) error(3.1415) range(15, 20) type(unsigned 3 4) error(1.6180) disabled final)");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  scalarMD = dyn_cast<mdutils::InputInfo>(parser.metadata.get());
  EXPECT_EQ(3, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getWidth());
  EXPECT_EQ(4, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->getPointPos());
  EXPECT_EQ(false, dyn_cast<mdutils::FPType>(scalarMD->IType.get())->isSigned());
  EXPECT_EQ(15, scalarMD->IRange->Min);
  EXPECT_EQ(20, scalarMD->IRange->Max);
  EXPECT_EQ(1.6180, *scalarMD->IError.get());
  EXPECT_EQ(false, scalarMD->IEnableConversion);
  EXPECT_EQ(true, scalarMD->IFinal);
}

TEST_F(AnnotationParserTest, ScalarInvalid)
{
  // not valid, random gibberish
  annstr = llvm::StringRef("scalar(random gibberish)");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_THAT(error, testing::StartsWith(UNK_ID_ERR));
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, ScalarDup)
{
  // not valid, duplicated data type pattern not admitted
  annstr = llvm::StringRef("scalar(range(0, 10) type(1 2) error(3.1415)) scalar(range(15, 20) type(unsigned 3 4) error(1.6180))");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(DUP_DTP_ERR, error);
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, ScalarAndStruct)
{
  // not valid, duplicated data type pattern not admitted
  annstr = llvm::StringRef("scalar() struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(DUP_DTP_ERR, error);
  ASSERT_FALSE(res);
}


/**
 * Valid struct syntax:
 * - struct[]
 * - struct[<data type pattern>]
 * - struct[<data type pattern>, ... <data type pattern>]
 *
 * To check the correctness of the parsing process, we use a scalar(error(<DTP>)) DTP
 * because we only need two checks to verify it and it allows us to discriminate
 * between different scalars.
 *
 */
TEST_F(AnnotationParserTest, StructVoid)
{
  // valid, void DTP
  annstr = llvm::StringRef("struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  structMD = dyn_cast<mdutils::StructInfo>(parser.metadata.get());
  EXPECT_EQ(1, structMD->size());
  EXPECT_EQ(nullptr, structMD->getField(0));
}

TEST_F(AnnotationParserTest, StructSingleScalar)
{
  // valid, single scalar DTP
  annstr = llvm::StringRef("struct[scalar(error(3.1415))]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  structMD = dyn_cast<mdutils::StructInfo>(parser.metadata.get());
  scalarMD = dyn_cast<mdutils::InputInfo>(structMD->getField(0).get());
  EXPECT_EQ(1, structMD->size());
  EXPECT_EQ(3.1415, *scalarMD->IError);
}

TEST_F(AnnotationParserTest, StructMultipleScalars)
{
  // valid, multiple scalar DTPs
  annstr = llvm::StringRef("struct[scalar(error(3.1415)), scalar(error(1.6180)), scalar(error(2.7183))]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  structMD = dyn_cast<mdutils::StructInfo>(parser.metadata.get());
  EXPECT_EQ(3, structMD->size());
  scalarMD = dyn_cast<mdutils::InputInfo>(structMD->getField(0).get());
  EXPECT_EQ(3.1415, *scalarMD->IError);
  scalarMD = dyn_cast<mdutils::InputInfo>(structMD->getField(1).get());
  EXPECT_EQ(1.6180, *scalarMD->IError);
  scalarMD = dyn_cast<mdutils::InputInfo>(structMD->getField(2).get());
  EXPECT_EQ(2.7183, *scalarMD->IError);
}

TEST_F(AnnotationParserTest, StructSingleStruct)
{
  // valid, single struct DTP
  annstr = llvm::StringRef("struct[struct[scalar(error(3.1415))]]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  structMD = dyn_cast<mdutils::StructInfo>(parser.metadata.get());
  EXPECT_EQ(1, structMD->size());
  structMD = dyn_cast<mdutils::StructInfo>(structMD->getField(0).get()); // inner struct
  EXPECT_EQ(1, structMD->size());
  scalarMD = dyn_cast<mdutils::InputInfo>(structMD->getField(0).get());
  EXPECT_EQ(3.1415, *scalarMD->IError);
}

TEST_F(AnnotationParserTest, StructMultipleStructs)
{
  // valid, multiple struct DTP
  annstr = llvm::StringRef("struct[struct[scalar(error(3.1415))], struct[scalar(error(1.6180)), scalar(error(2.7183))]]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  mdutils::StructInfo *innerStruct;
  structMD = dyn_cast<mdutils::StructInfo>(parser.metadata.get());
  EXPECT_EQ(2, structMD->size());
  innerStruct = dyn_cast<mdutils::StructInfo>(structMD->getField(0).get());
  EXPECT_EQ(1, innerStruct->size());
  scalarMD = dyn_cast<mdutils::InputInfo>(innerStruct->getField(0).get());
  EXPECT_EQ(3.1415, *scalarMD->IError);
  innerStruct = dyn_cast<mdutils::StructInfo>(structMD->getField(1).get());
  EXPECT_EQ(2, innerStruct->size());
  scalarMD = dyn_cast<mdutils::InputInfo>(innerStruct->getField(0).get());
  EXPECT_EQ(1.6180, *scalarMD->IError);
  scalarMD = dyn_cast<mdutils::InputInfo>(innerStruct->getField(1).get());
  EXPECT_EQ(2.7183, *scalarMD->IError);
}

TEST_F(AnnotationParserTest, StructMixedDTP)
{
  // valid, mixed DTPs
  annstr = llvm::StringRef("struct[scalar(error(3.1415)), struct[scalar(error(1.6180))]]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  structMD = dyn_cast<mdutils::StructInfo>(parser.metadata.get());
  EXPECT_EQ(2, structMD->size());
  scalarMD = dyn_cast<mdutils::InputInfo>(structMD->getField(0).get());
  EXPECT_EQ(3.1415, *scalarMD->IError);
  structMD = dyn_cast<mdutils::StructInfo>(structMD->getField(1).get()); // inner struct
  EXPECT_EQ(1, structMD->size());
  scalarMD = dyn_cast<mdutils::InputInfo>(structMD->getField(0).get());
  EXPECT_EQ(1.6180, *scalarMD->IError);
}

TEST_F(AnnotationParserTest, StructNoDTP)
{
  // not valid, no DTP
  annstr = llvm::StringRef("struct[]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(EMPTY_STRUCT_ERR, error);
  ASSERT_FALSE(res);
}


/**
 * Parsing integer throught backtracking(<int>) struct[void]
 * (the struct[void] is needed to avoid the missing DTP error,
 * it can be substituted with any other valid DTP)
 */
TEST_F(AnnotationParserTest, IntegerZero)
{
  // valid, positive integer without sign
  annstr = llvm::StringRef("backtracking(0) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, 0);
}

TEST_F(AnnotationParserTest, IntegerNoSign)
{
  // valid, positive integer without sign
  annstr = llvm::StringRef("backtracking(1234567890) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, 1234567890);
}

TEST_F(AnnotationParserTest, IntegerPositiveSign)
{
  // valid, positive integer
  annstr = llvm::StringRef("backtracking(+1234567890) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, 1234567890);
}

TEST_F(AnnotationParserTest, IntegerNegativeSign)
{
  // valid, negative integer
  annstr = llvm::StringRef("backtracking(-1234567890) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, -1234567890);
}

TEST_F(AnnotationParserTest, IntegerNoSignBase8)
{
  // valid, positive integer without sign in base 8
  // 1234567 (base 8) = 342391 (base 10)
  annstr = llvm::StringRef("backtracking(01234567) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, 342391);
}

TEST_F(AnnotationParserTest, IntegerPositiveSignBase8)
{
  // valid, positive integer in base 8
  // 1234567 (base 8) = 342391 (base 10)
  annstr = llvm::StringRef("backtracking(+01234567) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, 342391);
}

TEST_F(AnnotationParserTest, IntegerNegativeSignBase8)
{
  // valid, negative integer in base 8
  // 1234567 (base 8) = 342391 (base 10)
  annstr = llvm::StringRef("backtracking(-01234567) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, -342391);
}

TEST_F(AnnotationParserTest, IntegerNoSignBase16)
{
  // valid, positive integer without sign in base 16
  // 0x0123456789abcdef (base 16) = 81985529216486895 (base 10)
  annstr = llvm::StringRef("backtracking(0x0123456789AbCdEf) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, 81985529216486895);
}

TEST_F(AnnotationParserTest, IntegerPositiveSignBase16)
{
  // valid, positive integer in base 16
  // 0x0123456789abcdef (base 16) = 81985529216486895 (base 10)
  annstr = llvm::StringRef("backtracking(+0x0123456789aBcDeF) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, 81985529216486895);
}

TEST_F(AnnotationParserTest, IntegerNegativeSignBase16)
{
  // valid, negative integer in base 16
  // 0x0123456789abcdef (base 16) = 81985529216486895 (base 10)
  annstr = llvm::StringRef("backtracking(-0x0123456789abcDEF) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(parser.backtrackingDepth, -81985529216486895);
}

TEST_F(AnnotationParserTest, IntegerInvalidOctal)
{
  // not valid, base 8 number with digits >7
  annstr = llvm::StringRef("backtracking(0123456789) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, IntegerInvalidHex)
{
  // not valid, base 16 number with digits > f
  annstr = llvm::StringRef("backtracking(0x0123456789abG) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, IntegerInvalidReal)
{
  // not valid, real number
  annstr = llvm::StringRef("backtracking(-3.1415) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, IntegerInvalidString)
{
  // not valid, string
  annstr = llvm::StringRef("backtracking(\'+50\') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}


/**
 * Parsing strings through target(<string>) struct[void];
 * the struct[void] is needed to avoid the missing DTP error.
 */
TEST_F(AnnotationParserTest, StringEmpty)
{
  // valid, empty string
  annstr = llvm::StringRef("target('') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ("", parser.target.getValue());
}

TEST_F(AnnotationParserTest, StringAllLiterals)
{
  // valid, all printable characters except ' and @
  annstr = llvm::StringRef("target('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+,-./:;<=>?[]^_`{|}~\"\\') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+,-./:;<=>?[]^_`{|}~\"\\", parser.target.getValue());
}

TEST_F(AnnotationParserTest, StringEscapedQuote)
{
  // valid, with @'
  annstr = llvm::StringRef("target('@'') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ("'", parser.target.getValue());
}

TEST_F(AnnotationParserTest, StringEscapedAt)
{
  // valid, with @@
  annstr = llvm::StringRef("target('@@') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ("@", parser.target.getValue());
}

/*
 * Causes the whole test framework to hang indefinitely.
 * Such a condition however does not seem possible to occur.
 */
/*
TEST_F(AnnotationParserTest, StringUnmatchedQuote) {
 // not valid, unmatched quote
 annstr = llvm::StringRef("target('notvalid) struct[void]");
 res = parser.parseAnnotationString(annstr);
 error = parser.lastError().str();
 ASSERT_FALSE(res);
 EXPECT_EQ(UNK_ID_ERR, error);
}*/

TEST_F(AnnotationParserTest, StringUnescapedQuote)
{
  // not valid, quote not escaped
  annstr = llvm::StringRef("target('notvalid'') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, StringUnescapedAt)
{
  // not valid, @ not escaped
  annstr = llvm::StringRef("target('not@valid') struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, StringInvalidDoubeQuotes)
{
  // not valid, string enclosed in double quotes
  annstr = llvm::StringRef("target(\"notvalid\") struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}


/**
 * Parsing booleans through backtracking(<bool>) struct[void];
 * the struct[void] is needed to avoid the missing DTP error.
 */
TEST_F(AnnotationParserTest, BooleanTrue)
{
  // valid, true
  annstr = llvm::StringRef("backtracking(true) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(true, parser.backtracking);
}

TEST_F(AnnotationParserTest, BooleanFalse)
{
  // valid, false
  annstr = llvm::StringRef("backtracking(false) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(false, parser.backtracking);
}

TEST_F(AnnotationParserTest, BooleanYes)
{
  // valid, yes
  annstr = llvm::StringRef("backtracking(yes) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(true, parser.backtracking);
}

TEST_F(AnnotationParserTest, BooleanNo)
{
  // valid, no
  annstr = llvm::StringRef("backtracking(no) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  EXPECT_EQ(NO_ERR, error);
  ASSERT_TRUE(res);

  EXPECT_EQ(false, parser.backtracking);
}

TEST_F(AnnotationParserTest, BooleanInvalidTRUE)
{
  // not valid, uppercase true
  annstr = llvm::StringRef("backtracking(TRUE) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

TEST_F(AnnotationParserTest, BooleanInvalidFALSE)
{
  // not valid, uppercase false
  annstr = llvm::StringRef("backtracking(FALSE) struct[void]");
  res = parser.parseAnnotationString(annstr);
  error = parser.lastError().str();
  ASSERT_FALSE(res);
}

} // namespace