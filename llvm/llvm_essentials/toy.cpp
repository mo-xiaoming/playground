#include "llvm/IR/Instructions.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
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

llvm::Value* createIfElse(llvm::IRBuilder<>& builder, llvm::Function* func) {
  // add `%multmp = mul i32 %a, 16` after `entry`
  llvm::Value* arg1 = func->arg_begin();
  llvm::Value* constant = builder.getInt32(16);
  llvm::Value* val = createArith(builder, arg1, constant);

  // %cmptmp = icmp ult i32 %multmp, 100
  llvm::Value* cmpOp = builder.CreateICmpULT(val, builder.getInt32(100), "cmptmp");
  // %ifcond = icmp eq i1 %cmptmp, true
  llvm::Value* cond = builder.CreateICmpEQ(cmpOp, builder.getTrue(), "ifcond");

  // then:
  auto* thenBB = createBB(func, "then");
  // else:
  auto* elseBB = createBB(func, "else");
  // merge:
  auto* mergeBB = createBB(func, "merge");

  // br i1 %ifcond, label %then, label %else
	builder.CreateCondBr(cond, thenBB, elseBB);

  builder.SetInsertPoint(thenBB);
  // %thenret = add i32 %a, 1
  llvm::Value* thenVal = builder.CreateAdd(arg1, builder.getInt32(1), "thenret");
  // br label %merge
  builder.CreateBr(mergeBB);

  builder.SetInsertPoint(elseBB);
  // %elseret = add i32 %a, 2
  llvm::Value* elseVal = builder.CreateAdd(arg1, builder.getInt32(2), "elseret");
  // br label %merge
  builder.CreateBr(mergeBB);

  /// altogether we have a line `%iftmp = phi i32 [ %thenret, %then ], [ %elseret, %else ]`
  builder.SetInsertPoint(mergeBB);
  // %iftmp = phi i32
  auto* phi = builder.CreatePHI(builder.getInt32Ty(), 2, "iftmp");
  // [ %thenret, %then ]
  phi->addIncoming(thenVal, thenBB);
  // [ %elseret, %else ]
  phi->addIncoming(elseVal, elseBB);

  return phi;
}

llvm::Value* createLoop(llvm::Module& module, llvm::IRBuilder<>& builder) {
  // nothing
  auto* func = createFunc(module, builder, "bar");

  // declaration with entry label
  auto* entry = createBB(func, "entry");
  builder.SetInsertPoint(entry);

  auto* it = func->arg_begin();
  llvm::Value* a = it;
  llvm::Value* b = it + 1;
  llvm::Value* constant = builder.getInt32(16);
  llvm::Value* val = createArith(builder, a, constant);

  // loop:
  llvm::BasicBlock* loopBB = createBB(func, "loop");
  // afterloop:
  llvm::BasicBlock* afterBB = createBB(func, "afterloop");

  llvm::Value* startVal = builder.getInt32(1);
  llvm::BasicBlock* preHeaderBB = builder.GetInsertBlock();
  builder.CreateBr(loopBB);
  builder.SetInsertPoint(loopBB);
  llvm::PHINode* indVar = builder.CreatePHI(builder.getInt32Ty(), 2, "i");
  indVar->addIncoming(startVal, preHeaderBB);

  llvm::Value* add = builder.CreateAdd(a, builder.getInt32(5), "addtmp");
  llvm::Value* stepVal = builder.getInt32(1);
  llvm::Value* nextVal = builder.CreateAdd(indVar, stepVal, "nextval");
  llvm::Value* endCond = builder.CreateICmpULT(indVar, b, "endcond");
  endCond = builder.CreateICmpEQ(endCond, builder.getTrue(), "loopcond");
  llvm::BasicBlock* loopEndBB = builder.GetInsertBlock();
  builder.CreateCondBr(endCond, loopBB, afterBB);
  builder.SetInsertPoint(afterBB);
  indVar->addIncoming(nextVal, loopEndBB);

  builder.CreateRet(add);
  return func;
}

int main() {
  // only generates below two lines
  // ; ModuleID = 'my compiler'
  // source_filename = "my compiler"
  auto module = std::make_unique<llvm::Module>("my compiler", context());

  llvm::IRBuilder<> builder(context());
  // auto builder{llvm::IRBuilder<>(context())};

  auto* gVar = createGlob(*module, builder, "x");

  auto* func = createFunc(*module, builder, "foo");

  auto* entry = createBB(func, "entry");
  builder.SetInsertPoint(entry);

  llvm::Value* v = createIfElse(builder, func);

  // add `ret int32 0` as terminator instruction
  // builder.CreateRet(builder.getInt32(0));

  // `ret i32 %iftmp` to the end of `foo`
  builder.CreateRet(v);

  auto* ret2 = createLoop(*module, builder);

  llvm::verifyFunction(*func);

  llvm::verifyModule(*module);
  module->print(llvm::outs(), nullptr);
}
