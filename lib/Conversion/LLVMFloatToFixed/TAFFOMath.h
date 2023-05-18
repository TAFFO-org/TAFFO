#pragma once
#include "FixedPointType.h"
#include "LLVMFloatToFixedPass.h"
#include "SinCos.h"
#include "TypeUtils.h"
#include "string"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>

using namespace flttofix;
using namespace llvm;
using namespace taffo;


namespace taffo
{
inline bool start_with(llvm::Twine str, llvm::Twine prfx)
{
  return str.getSingleStringRef().startswith(prfx.getSingleStringRef());
}

inline llvm::Type *getElementTypeFromValuePointer(llvm::Value *pointer_value)
{
  return cast<PointerType>(pointer_value->getType())->getPointerElementType();
}

} // namespace taffo

// value taken from Elementary Function Chapter 7. The CORDIC Algorithm
namespace TaffoMath
{

constexpr int TABLELENGHT = 64;
const double arctan_2power[TABLELENGHT] = {0.785398163397448309615660845819875721049292349843776455243736148076954101571552249657008706335529267, 0.4636476090008061162142562314612144020285370542861202638109330887201978641657417053006002839848878926, 0.2449786631268641541720824812112758109141440983811840671273759146673551195876420965745341576687019914, 0.1243549945467614350313548491638710255731701917698040899151141191157222674275667586237105943133533303, 0.06241880999595734847397911298550511360627388779749919460752781689869026721680345781396936172340611606, 0.03123983343026827625371174489249097703249566372540004025531558625579642101943244717896297220075854501, 0.0156237286204768308028015212565703189111141398009054178814105073966647741764017791232644450551360346, 0.007812341060101111296463391842199281621222811725014723557453902248389872045335231379302642040135812015, 0.003906230131966971827628665311424387140357490115202856215213095149011344163954380208495709022456100032, 0.001953122516478818685121482625076713931610746777233510339057533960431085303137097972915916496709085832, 0.0009765621895593194304034301997172908516341970158100875900490072522676375203550845442358156030277274715, 0.0004882812111948982754692396256448486661923611331350030371094033534875121367432795866327618431818746773, 0.0002441406201493617640167229432596599862124177909706176118079004609101784702174630533398957563177213556, 0.0001220703118936702042390586461179563009308294090157874984519398378466425902204557736661063332274582931, 0.00006103515617420877502166256917382915378514353683334617933767113431658656577688980710603690915222646626, 0.00003051757811552609686182595343853601975094967511943783753102115688361163048611109467083048692072421563, 0.00001525878906131576210723193581269788513742923814457587484624118640744586426707683086018832544324023914, 0.000007629394531101970263388482340105090586350743918468077157763830696533368540109726685013870065074828268, 0.000003814697265606496282923075616372993722805257303968866310187439250393888463610412929161242262088106293, 0.000001907348632810187035365369305917244168714342165450153366670057723467064463709843215578674212783051631, 0.0000009536743164059608794206706899231123900196341244987901601336118020760033298881120340466942799229875223, 0.0000004768371582030888599275838214492470758704940437866419674005321588714236381444304017560788393492299195, 0.0000002384185791015579824909479772189326978309689876906315591376691137221764828210301969320598648727962916, 0.0000001192092895507806853113684971379221126459675876645867355767382252154371995889555916850877159668978201, 0.00000005960464477539055441392106214178887425003019578236629731429456571000510846165866317908067872814110343, 0.00000002980232238769530367674013276770950334904390706744510724925847784084355726084717708524030644266201337, 0.00000001490116119384765514709251659596324710824893002596472001217005780549101420672737709864545287505378661, 0.000000007450580596923827987136564574495392113206692554566587007594760141617371183698194764539981295353052542, 0.000000003725290298461914045267070571811923583671948328737040524231998269239107397435819612345659409983754584, 0.000000001862645149230957029095883821476490434506528283573886351349105012495130259443092823157958078737120733, 0.0000000009313225746154785153557354776845613038929264961492906739437685424219745532957262514553181772562186441, 0.0000000004656612873077392577788419347105701629734786389156161742132349255441464969391566475332407218389649913, 0.0000000002328306436538696289020427418388212703712742932049818605254866622806071463876315999608347025264367517, 0.0000000001164153218269348144525990927298526587963964573800142900265849791708846857314265244701935173633649274, 0.00000000005820766091346740722649676159123158234954915625779527242397620616714716236559954724829144139773906282, 0.00000000002910383045673370361327303269890394779369363200363983049582993452502914809634311344424886073556054517, 0.00000000001455191522836685180663959783736299347421170360893671073206727021330709416425327393990286506261971405, 0.000000000007275957614183425903320184104670374184276462938882142964011175289083898575110032873215688950294386567, 0.000000000003637978807091712951660140200583796773034557866977925811829608364648574098763834528135623479457604381, 0.000000000001818989403545856475830076118822974596629319733360292537145207653503355300552372502883812834006560898, 0.0000000000009094947017729282379150388117278718245786649666696631862264792881854907698288678102920473858647550416, 0.0000000000004547473508864641189575194999034839780723331208336962301246639213824854511171252014823517399223410002, 0.000000000000227373675443232059478759761706685497259041640104211664135781552996538778417204823716520471369361356, 0.0000000000001136868377216160297393798823227106871573802050130264466222913992128088503342604054645163271838561437, 0.00000000000005684341886080801486968994134502633589467252562662830547170263443560865326153596968366536521881958807, 0.00000000000002842170943040400743484497069547204198683406570332853817283521085238881750049679045791817568728197127, 0.00000000000001421085471520200371742248535060588024835425821291606727125666327992165643264974862672048484696197641, 0.000000000000007105427357601001858711242675661672531044282276614508408896216095095614999240207635192954662689911913, 0.000000000000003552713678800500929355621337875677816380535284576813551111687423921495873191244362425726348511310925, 0.000000000000001776356839400250464677810668943444102047566910572101693888950315866266484095349870554046904025630625, 0.0000000000000008881784197001252323389053344724227002559458638215127117361184578544107948852451189833443324021783421, 0.0000000000000004440892098500626161694526672362989312819932329776890889670147968683990832473220894092958061970921672, 0.0000000000000002220446049250313080847263336181604132852491541222111361208768492846935645898735877241737809196785632, 0.0000000000000001110223024625156540423631668090815750981561442652763920151096061504661855482328961701470915257410477, 0.00000000000000005551115123125782702118158340454095860601951803315954900188870076849200725523219632453792921917953387, 0.00000000000000002775557561562891351059079170227050068512743975414494362523608759605161759633224592504441436422437612, 0.00000000000000001387778780781445675529539585113525301532842996926811795315451094950614334608616265889546345839763871, 0.00000000000000000693889390390722837764769792556762684175980374615851474414431386868825826659029007980771142176437958, 0.000000000000000003469446951953614188823848962783813462641850468269814343018039233586031981709083752083644505516318533, 0.000000000000000001734723475976807094411924481391906736541168808533726792877254904198253988288176015638820581245657663, 0.0000000000000000008673617379884035472059622406959533689231148510667158491096568630247817482414763940369889794699608906, 0.0000000000000000004336808689942017736029811203479766845431237313833394811387071078780977185209799990071903851466905389, 0.000000000000000000216840434497100886801490560173988342281757653922917435142338388484762214814834857680666509478115862, 0.0000000000000000001084202172485504434007452800869941711421533004903646793927922985605952768518453683914823046639763435}; // the first 64 iteration of arctan(2^-n) with n
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       // from 0 to 64
const double K = 1.646760258121065587423679402159867312986265495165396570148627753176701278048460842259042658967080647;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                // scale factor
const double Kopp = 0.607252935008881256169446752504928263112390852150089772456976013110147881208425;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  // 1/k
const double pi_half = 1.570796326794896619231321691639751442098584699687552910487472296153908203143104499314017412671058534;
const double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068;
const double pi_32 = 4.712388980384689857693965074919254326295754099062658731462416888461724609429313497942052238013175602;
const double pi_2 = 6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136;
const double zero = 0.0f;
const double one = 1.0f;
const double minus_one = -1.0f;


/** used to couple fixedpoint to corresponding value
 * @param T lenght of array
 * @param U type of llvm to couple
 */
template <typename U, int T = 0>
struct pair_ftp_value {
  SmallVector<flttofix::FixedPointType, T> fpt;
  SmallVector<U, T> value;
  pair_ftp_value() {}
  pair_ftp_value(const SmallVector<flttofix::FixedPointType, T> &fpt) : fpt(fpt), value() {}
  ~pair_ftp_value() {}
};
/** partial specialization for T=0
 * @param T lenght of array
 * @param U type of llvm to couple
 */
template <typename U>
struct pair_ftp_value<U, 0> {
  flttofix::FixedPointType fpt;
  U value;
  pair_ftp_value() {}
  pair_ftp_value(const flttofix::FixedPointType &fpt) : fpt(fpt), value() {}
  ~pair_ftp_value() {}
};

bool createFixedPointFromConst(
    llvm::LLVMContext &cont, FloatToFixed *ref, const double &current_float,
    const FixedPointType &match, llvm::Constant *&outI, FixedPointType &outF);


/**
 * @param ref used to access member function
 * @param oldf function used to take ret information
 * @param fxpret output of the function *
 * @param n specify wich argument return, valid values ranges from 0 to max number of argument
 * @param found used to return if information was found
 * */
void getFixedFromArg(FloatToFixed *ref, Function *oldf,
                     FixedPointType &fxparg, int n, bool &found);
/**
 * @param ref used to access member function
 * @param oldf function used to take ret information
 * @param fxpret output of the function *
 * @param found used to return if information was found
 * */
void getFixedFromRet(FloatToFixed *ref, Function *oldf,
                     FixedPointType &fxpret, bool &found);


llvm::GlobalVariable *
createGlobalConst(llvm::Module *module, llvm::StringRef Name, llvm::Type *Ty,
                  Constant *initializer, unsigned int alignment = 1);


Value *addAllocaToStart(FloatToFixed *ref, Function *oldf,
                        llvm::IRBuilder<> &builder, Type *to_alloca,
                        llvm::Value *ArraySize = (llvm::Value *)nullptr,
                        const llvm::Twine &Name = "");


} // namespace TaffoMath





namespace flttofix
{
bool partialSpecialCall(llvm::Function *newf, bool &foundRet, flttofix::FixedPointType &fxpret);
}