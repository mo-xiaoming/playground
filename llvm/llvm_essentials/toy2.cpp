#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>

llvm::LLVMContext& context() {
  static llvm::LLVMContext Context;
  return Context;
}

llvm::Function* createFunc(llvm::Module& module, llvm::IRBuilder<>& builder,
                           std::string const& name) {
  // only generates declaration
  // declare i32 @foo()
  auto* funcType = llvm::FunctionType::get(builder.getInt32Ty(), false);
  return llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name, module);
}

int main() {
  // only generates below two lines
  // ; ModuleID = 'my compiler'
  // source_filename = "my compiler"
  auto module0b = std::make_unique<llvm::Module>("my compiler", context());

  auto builder = llvm::IRBuilder<>(context());
  auto* func = createFunc(*module0b, builder, "foo");
  llvm::verifyFunction(*func);

  llvm::verifyModule(*module0b);
  module0b->print(llvm::outs(), nullptr);
}
