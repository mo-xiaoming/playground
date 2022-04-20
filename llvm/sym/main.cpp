#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

LLVMContext ctx;

std::unique_ptr<Module> createModule(char const* name) {
  auto mod = std::make_unique<Module>(name, ctx);

  auto c = mod->getOrInsertFunction("mul_add", Type::getInt32Ty(ctx), Type::getInt32Ty(ctx),
                                    Type::getInt32Ty(ctx), Type::getInt32Ty(ctx));

  auto* mul_add = cast<Function>(c.getCallee());
  mul_add->setCallingConv(CallingConv::C);

  Function::arg_iterator args = mul_add->arg_begin();
  Value* x = args++;
  x->setName("x");
  Value* y = args++;
  y->setName("y");
  Value* z = args++;
  z->setName("z");

  BasicBlock* block = BasicBlock::Create(ctx, "entry", mul_add);
  IRBuilder<> builder(block);

  Value* tmp = builder.CreateBinOp(Instruction::Mul, x, y, "tmp");
  Value* tmp2 = builder.CreateBinOp(Instruction::Add, tmp, z, "tmp2");

  builder.CreateRet(tmp2);

  return mod;
}

ExecutionEngine* createEE(std::unique_ptr<Module> m) {
  std::string err;
  auto* ee = EngineBuilder(std::move(m)).setErrorStr(&err).create();
  if (ee == nullptr) {
    errs() << "error: " << err << '\n';
    return nullptr;
  }
  ee->finalizeObject();
  if (!err.empty()) {
    errs() << "error: " << err << '\n';
    return nullptr;
  }
  return ee;
};

int main() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
  LLVMLinkInMCJIT();

  std::string err;
  auto m1 = createModule("module1");
  m1->print(llvm::errs(), nullptr, false, true);
  auto* e1 = createEE(std::move(m1));
  assert(e1 != nullptr);
  errs() << "from module1 " << e1->FindFunctionNamed("mul_add") << '\n';

  errs() << "--------------------------------------------------------------\n";

  auto m2 = createModule("module2");
  m2->print(llvm::errs(), nullptr, false, true);
  auto* e2 = createEE(std::move(m2));
  assert(e2 != nullptr);
  errs() << "from module2 " << e2->FindFunctionNamed("mul_add") << '\n';
}
