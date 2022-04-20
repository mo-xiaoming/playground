#include "codegen.h"
#include "node.h"
#include <iostream>
#include <llvm-10/llvm/ADT/APInt.h>
#include <llvm-10/llvm/IR/Constants.h>
#include <llvm-10/llvm/Support/raw_ostream.h>

using namespace std;

extern int yyparse();
extern NBlock* programBlock;

llvm::Function* createPrintfFunction(CodeGenContext& context) {
  std::vector<llvm::Type*> printf_arg_types{llvm::Type::getInt8PtrTy(myContext())}; // char*
  llvm::FunctionType* printf_type =
      llvm::FunctionType::get(context.builder.getInt32Ty(), printf_arg_types, true);
  return llvm::Function::Create(printf_type, llvm::Function::ExternalLinkage, "printf",
                                context.module.get());
}

void createEchoFunction(CodeGenContext& context, llvm::Function* printfFn) {
  std::vector<llvm::Type*> echo_arg_types{llvm::Type::getInt64Ty(myContext())};
  llvm::FunctionType* echo_type =
      llvm::FunctionType::get(context.builder.getInt32Ty(), echo_arg_types, false);
  llvm::Function* func = llvm::Function::Create(echo_type, llvm::Function::InternalLinkage,
                                                "echo", context.module.get());

  llvm::BasicBlock* bblock = llvm::BasicBlock::Create(myContext(), "entry", func);
  context.builder.SetInsertPoint(bblock);
  context.pushBlock(bblock);

#if 1
  Value* fmtStr = context.builder.CreateGlobalStringPtr("%ld\n");
#else
  const char* constValue = "%d\n";
  llvm::Constant* format_const = llvm::ConstantDataArray::getString(myContext(), constValue);
  auto* var = new llvm::GlobalVariable(
      *context.module,
      llvm::ArrayType::get(llvm::Type::getInt8Ty(myContext()), strlen(constValue) + 1), true,
      llvm::GlobalValue::PrivateLinkage, format_const, ".str");
  llvm::Constant* zero = llvm::Constant::getNullValue(llvm::IntegerType::getInt32Ty(myContext()));

  std::vector<llvm::Constant*> indices;
  indices.push_back(zero);
  indices.push_back(zero);
  llvm::Constant* var_ref = llvm::ConstantExpr::getGetElementPtr(
      llvm::ArrayType::get(llvm::Type::getInt8Ty(myContext()), strlen(constValue) + 1), var,
      indices);
#endif

  auto& fst_arg = *func->arg_begin();
  auto args = std::vector<Value*>{fmtStr, &fst_arg};

  context.builder.CreateCall(context.module->getFunction("printf"), args);
  context.builder.CreateRet(llvm::ConstantInt::get(myContext(), llvm::APInt(32, 13)));
#if 0
  CallInst* call = CallInst::Create(printfFn, args, "", bblock);
  ReturnInst::Create(myContext(), bblock);
#endif
  context.popBlock();

  //context.module->print(llvm::outs(), nullptr);
}

void createCoreFunctions(CodeGenContext& context) {
  llvm::Function* printfFn = createPrintfFunction(context);
  createEchoFunction(context, printfFn);
}
