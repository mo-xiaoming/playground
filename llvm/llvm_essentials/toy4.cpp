#include <llvm-10/llvm/ADT/ArrayRef.h>
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
#if 0
  // declare i32 @foo()
  auto* funcType = llvm::FunctionType::get(builder.getInt32Ty(), false); // no argument
#endif
  // define i32 @foo(i32 %0, i32 %1)
  constexpr auto argNames = std::array<char const*, 2>{"a", "b"};

  auto const argTys =
      llvm::SmallVector<llvm::Type*, argNames.size()>(argNames.size(), builder.getInt32Ty());
  auto* funcType = llvm::FunctionType::get(builder.getInt32Ty(), argTys, false); // with argument
  auto* func = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name, module);

  // define i32 @foo(i32 %a, i32 %b)
  std::size_t i = 0;
  for (auto& arg : func->args()) {
    arg.setName(argNames[i]);
    ++i;
  }

  return func;
}

llvm::BasicBlock* createBB(llvm::Function* func, llvm::StringRef name) {
  // append `{ entry: }` into function `foo`
  return llvm::BasicBlock::Create(context(), name, func);
}

llvm::GlobalVariable* createGlob(llvm::Module& module, llvm::IRBuilder<>& builder,
                                 llvm::StringRef name) {
  // @x = common global i32, align 4
  module.getOrInsertGlobal(name, builder.getInt32Ty());
  llvm::GlobalVariable* var = module.getNamedGlobal(name);
  var->setLinkage(llvm::GlobalVariable::CommonLinkage);
  var->setAlignment(llvm::Align(4));
  return var;
}

llvm::Value* createArith(llvm::IRBuilder<>& builder, llvm::Value* lhs, llvm::Value* rhs) {
  return builder.CreateMul(lhs, rhs, "multmp");
}

int main() {
  // only generates below two lines
  // ; ModuleID = 'my compiler'
  // source_filename = "my compiler"
  auto module0b = std::make_unique<llvm::Module>("my compiler", context());

  auto builder = llvm::IRBuilder<>(context());

  auto* gVar = createGlob(*module0b, builder, "x");

  auto* func = createFunc(*module0b, builder, "foo");

  auto* entry = createBB(func, "entry");
  builder.SetInsertPoint(entry);

	// add `%multmp = mul i32 %a, 16` after `entry`
  llvm::Value* arg1 = func->arg_begin();
  llvm::Value* constant = builder.getInt32(16);
  llvm::Value* val = createArith(builder, arg1, constant);

  // add `ret int32 0` as terminator instruction
  builder.CreateRet(builder.getInt32(0));

  llvm::verifyFunction(*func);

  llvm::verifyModule(*module0b);
  module0b->print(llvm::outs(), nullptr);
}
