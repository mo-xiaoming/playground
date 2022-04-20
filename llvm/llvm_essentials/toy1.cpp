#include <llvm-10/llvm/IR/Attributes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

llvm::LLVMContext& context() {
  static llvm::LLVMContext Context;
  return Context;
}

int main(int argc, char** argv) {
  auto Module0b = std::make_unique<llvm::Module>("my compiler", context());
  Module0b->print(llvm::outs(), nullptr);
}
