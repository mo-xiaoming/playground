/*
*/

#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

int main() {
  LLVMContext ctx;
  Module mod0("module0", ctx);
  Function::Create(FunctionType::get(Type::getDoubleTy(ctx), Type::getVoidTy(ctx), false),
                   Function::ExternalLinkage, "test", mod0);
  Module mod1("module1", ctx);
  Function::Create(FunctionType::get(Type::getDoubleTy(ctx), Type::getVoidTy(ctx), false),
                   Function::ExternalLinkage, "test", mod1);
	mod0.print(llvm::errs(), nullptr, false, true);
	mod1.print(llvm::errs(), nullptr, false, true);
}
