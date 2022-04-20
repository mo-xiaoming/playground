#include <llvm-10/llvm/ADT/StringRef.h>
#include <llvm-10/llvm/IR/BasicBlock.h>
#include <llvm-10/llvm/IR/GlobalVariable.h>
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

llvm::Function* createFunc(llvm::Module& module, llvm::IRBuilder<>& builder, llvm::StringRef name) {
  auto* funcType = llvm::FunctionType::get(builder.getInt32Ty(), false);
  return llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name, module);
}

llvm::BasicBlock* createBB(llvm::Function* func, llvm::StringRef name) {
  return llvm::BasicBlock::Create(context(), name, func);
}

llvm::GlobalVariable* createGlob(llvm::Module& module, llvm::IRBuilder<>& builder,
                                 llvm::StringRef name) {
  module.getOrInsertGlobal(name, builder.getInt32Ty());
  llvm::GlobalVariable* var = module.getNamedGlobal(name);
  var->setLinkage(llvm::GlobalVariable::CommonLinkage);
  var->setAlignment(llvm::Align(4));
  return var;
}

int main() {
  // only generates below two lines
  // ; ModuleID = 'my compiler'
  // source_filename = "my compiler"
  auto module0b = std::make_unique<llvm::Module>("my compiler", context());

  auto builder = llvm::IRBuilder<>(context());

	// @x = common global i32, align 4
	auto* g_var = createGlob(*module0b, builder, "x");

  // declare i32 @foo()
  auto* func = createFunc(*module0b, builder, "foo");

  // append `{ entry: }` into function `foo`
  auto* entry = createBB(func, "entry");
  builder.SetInsertPoint(entry);

	// add `ret int32 0` as terminator instruction
	builder.CreateRet(builder.getInt32(0));

  llvm::verifyFunction(*func);

  llvm::verifyModule(*module0b);
  module0b->print(llvm::outs(), nullptr);
}
