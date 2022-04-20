/*
; ModuleID = 'module0'
source_filename = "test"

declare double @test(void %0)

declare double @test.1(void %0)
*/

#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

int main() {
  LLVMContext ctx;
  Module mod("module0", ctx);
  GlobalVariable(Type::getDoubleTy(ctx), true, GlobalValue::ExternalLinkage, nullptr, "test");
  GlobalVariable(Type::getDoubleTy(ctx), true, GlobalValue::ExternalLinkage, nullptr, "test");
	mod.print(llvm::errs(), nullptr, false, true);
}
