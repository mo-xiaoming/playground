#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

orc::ThreadSafeContext tsc(std::make_unique<LLVMContext>());

orc::ThreadSafeModule createTSM(char const* name) {
  auto* ctx = tsc.getContext();
  auto mod = std::make_unique<Module>(name, *ctx);

  auto c = mod->getOrInsertFunction("mul_add", Type::getInt32Ty(*ctx), Type::getInt32Ty(*ctx),
                                    Type::getInt32Ty(*ctx), Type::getInt32Ty(*ctx));

  auto* mul_add = cast<Function>(c.getCallee());
  mul_add->setCallingConv(CallingConv::C);

  Function::arg_iterator args = mul_add->arg_begin();
  Value* x = args++;
  x->setName("x");
  Value* y = args++;
  y->setName("y");
  Value* z = args++;
  z->setName("z");

  BasicBlock* block = BasicBlock::Create(*ctx, "entry", mul_add);
  IRBuilder<> builder(block);

  Value* tmp = builder.CreateBinOp(Instruction::Mul, x, y, "tmp");
  Value* tmp2 = builder.CreateBinOp(Instruction::Add, tmp, z, "tmp2");

  builder.CreateRet(tmp2);

  return orc::ThreadSafeModule(std::move(mod), tsc);
}

int main() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();

  auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
  cantFail(jtmb.takeError());

  auto targetMachine = jtmb->createTargetMachine();
  cantFail(targetMachine.takeError());

  auto dataLayout = jtmb->getDefaultDataLayoutForTarget();
  cantFail(dataLayout.takeError());

  orc::ExecutionSession execSession;

  orc::MangleAndInterner mangle(execSession, *dataLayout);

  orc::RTDyldObjectLinkingLayer objectLayer(
      execSession, [] { return std::make_unique<llvm::SectionMemoryManager>(); });
  orc::IRCompileLayer compileLayer(execSession, objectLayer,
                                   std::make_unique<orc::SimpleCompiler>(*targetMachine.get()));

  orc::JITDylib& mainJD(execSession.createBareJITDylib("main"));
  mainJD.addGenerator(cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      dataLayout->getGlobalPrefix())));

  const auto addModule = [&](orc::ThreadSafeModule tsm) {
    tsm.withModuleDo([&](llvm::Module& m) {
      m.setDataLayout(*dataLayout);
      cantFail(compileLayer.add(mainJD, std::move(tsm)));
    });
  };
  const auto lookup = [&](StringRef name) -> Expected<JITEvaluatedSymbol>  {
    return execSession.lookup({&mainJD}, mangle(name.str()));
  };

  addModule(createTSM("module1"));
  addModule(createTSM("module2"));

  auto sym = cantFail(lookup("mul_add"));
  errs() << jitTargetAddressToPointer<void*>(sym.getAddress()) << '\n';
}
