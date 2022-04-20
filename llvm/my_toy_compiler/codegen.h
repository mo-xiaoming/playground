#include <llvm/Bitstream/BitstreamReader.h>
#include <llvm/Bitstream/BitstreamWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <stack>
#include <typeinfo>

using namespace llvm;

class NBlock;

#if 1
inline LLVMContext& myContext() {
  static LLVMContext MyContext;
  return MyContext;
}
#else

static LLVMContext llvmContext;
#define myContext() llvmContext

#endif

class CodeGenContext {
  struct CodeGenBlock {
    BasicBlock* block = nullptr;
    Value* returnValue = nullptr;
    std::map<std::string, Value*> locals;
  };

  std::stack<CodeGenBlock*> blocks;
  Function* mainFunction = nullptr;

public:
  IRBuilder<> builder{myContext()};
  std::unique_ptr<Module> module{std::make_unique<Module>("main", myContext())};

  void generateCode(NBlock& root);
  GenericValue runCode();
  std::map<std::string, Value*>& locals() { return blocks.top()->locals; }
  BasicBlock* currentBlock() { return blocks.top()->block; }
  void pushBlock(BasicBlock* block) {
    blocks.push(new CodeGenBlock());
    blocks.top()->block = block;
  }
  void popBlock() {
    CodeGenBlock* top = blocks.top();
    blocks.pop();
    delete top;
  }
  void setCurrentReturnValue(Value* value) { blocks.top()->returnValue = value; }
  Value* getCurrentReturnValue() { return blocks.top()->returnValue; }
};
