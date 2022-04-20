#include "codegen.h"
#include "node.h"
#include "parser.hpp"

using namespace std;

/* Compile the AST into a module */
void CodeGenContext::generateCode(NBlock& root) {
  std::cout << "Generating code...\n";

  /* Create the top level interpreter function to call as entry */
  vector<Type*> argTypes;
  FunctionType* ftype =
      FunctionType::get(Type::getInt32Ty(myContext()), makeArrayRef(argTypes), false);
  mainFunction = Function::Create(ftype, GlobalValue::InternalLinkage, "main", module.get());
  BasicBlock* bblock = BasicBlock::Create(myContext(), "entry", mainFunction, nullptr);

  /* Push a new variable/block context */
  pushBlock(bblock);
  root.codeGen(*this); /* emit bytecode for the toplevel block */
  ReturnInst::Create(myContext(), bblock);
  popBlock();

  /* Print the bytecode in a human-readable format
     to see if our program compiled properly
   */
  std::cout << "Code is generated.\n";

  legacy::PassManager pm;
  pm.add(createPrintModulePass(outs()));
  pm.run(*module);
}

/* Executes the AST by running the main function */
GenericValue CodeGenContext::runCode() {
  std::cout << "Running code...\n";
  ExecutionEngine* ee = EngineBuilder(std::move(module)).create();
  ee->finalizeObject();

#if 0
  auto f = reinterpret_cast<int(*)()>(ee->getFunctionAddress("main"));
  int result = f();
  builder.CreateRet(llvm::ConstantInt::get(myContext(), llvm::APInt(32, 7)));
  std::cout << "main returns " << result << '\n';
  return {};
#else
  vector<GenericValue> noargs;
  GenericValue v = ee->runFunction(mainFunction, noargs);
  std::cout << "Code was run.\n";
  return v;
#endif
}

/* Returns an LLVM type based on the identifier */
static Type* typeOf(const NIdentifier& type) {
  if (type.name == "int") {
    return Type::getInt64Ty(myContext());
  }
  if (type.name == "double") {
    return Type::getDoubleTy(myContext());
  }
  return Type::getVoidTy(myContext());
}

/* -- Code Generation -- */

Value* NInteger::codeGen(CodeGenContext& /*context*/) {
  std::cout << "Creating integer: " << value << endl;
  return ConstantInt::get(Type::getInt64Ty(myContext()), value, true);
}

Value* NDouble::codeGen(CodeGenContext& /*context*/) {
  std::cout << "Creating double: " << value << endl;
  return ConstantFP::get(Type::getDoubleTy(myContext()), value);
}

Value* NIdentifier::codeGen(CodeGenContext& context) {
  std::cout << "Creating identifier reference: " << name << endl;
  if (context.locals().find(name) == context.locals().end()) {
    std::cerr << "undeclared variable " << name << endl;
    return nullptr;
  }
  return new LoadInst(context.locals()[name], "", false, context.currentBlock());
}

Value* NMethodCall::codeGen(CodeGenContext& context) {
  Function* function = context.module->getFunction(id.name);
  if (function == nullptr) {
    std::cerr << "no such function " << id.name << endl;
  }

std::cout << id.name << ": " << function->getName().str() << '\n';

  std::vector<Value*> args;
  ExpressionList::const_iterator it;
  for (it = arguments.begin(); it != arguments.end(); it++) {
    args.push_back((**it).codeGen(context));
  }
  CallInst* call = CallInst::Create(function, args, "", context.currentBlock());
  std::cout << "Creating method call: " << id.name << endl;
  return call;
}

Value* NBinaryOperator::codeGen(CodeGenContext& context) {
  std::cout << "Creating binary operation " << op << endl;
  Instruction::BinaryOps instr;
  switch (op) {
  case TPLUS:
    instr = Instruction::Add;
    goto math;
  case TMINUS:
    instr = Instruction::Sub;
    goto math;
  case TMUL:
    instr = Instruction::Mul;
    goto math;
  case TDIV:
    instr = Instruction::SDiv;
    goto math;

    /* TODO comparison */
  }

  return nullptr;
math:
  return BinaryOperator::Create(instr, lhs.codeGen(context), rhs.codeGen(context), "",
                                context.currentBlock());
}

Value* NAssignment::codeGen(CodeGenContext& context) {
  std::cout << "Creating assignment for " << lhs.name << endl;
  if (context.locals().find(lhs.name) == context.locals().end()) {
    std::cerr << "undeclared variable " << lhs.name << endl;
    return nullptr;
  }
  return new StoreInst(rhs.codeGen(context), context.locals()[lhs.name], false,
                       context.currentBlock());
}

Value* NBlock::codeGen(CodeGenContext& context) {
  StatementList::const_iterator it;
  Value* last = nullptr;
  for (it = statements.begin(); it != statements.end(); it++) {
    std::cout << "Generating code for " << typeid(**it).name() << endl;
    last = (**it).codeGen(context);
  }
  std::cout << "Creating block" << endl;
  return last;
}

Value* NExpressionStatement::codeGen(CodeGenContext& context) {
  std::cout << "Generating code for " << typeid(expression).name() << endl;
  return expression.codeGen(context);
}

Value* NReturnStatement::codeGen(CodeGenContext& context) {
  std::cout << "Generating return code for " << typeid(expression).name() << endl;
  Value* returnValue = expression.codeGen(context);
  context.setCurrentReturnValue(returnValue);
  return returnValue;
}

Value* NVariableDeclaration::codeGen(CodeGenContext& context) {
  std::cout << "Creating variable declaration " << type.name << " " << id.name << endl;
  auto* alloc = new AllocaInst(typeOf(type), 0, id.name, context.currentBlock());
  context.locals()[id.name] = alloc;
  if (assignmentExpr != nullptr) {
    NAssignment assn(id, *assignmentExpr);
    assn.codeGen(context);
  }
  return alloc;
}

Value* NExternDeclaration::codeGen(CodeGenContext& context) {
  vector<Type*> argTypes;
  VariableList::const_iterator it;
  for (it = arguments.begin(); it != arguments.end(); it++) {
    argTypes.push_back(typeOf((**it).type));
  }
  FunctionType* ftype = FunctionType::get(typeOf(type), makeArrayRef(argTypes), false);
  Function* function =
      Function::Create(ftype, GlobalValue::ExternalLinkage, id.name, context.module.get());
  return function;
}

Value* NFunctionDeclaration::codeGen(CodeGenContext& context) {
  vector<Type*> argTypes;
  VariableList::const_iterator it;
  for (it = arguments.begin(); it != arguments.end(); it++) {
    argTypes.push_back(typeOf((**it).type));
  }
  FunctionType* ftype = FunctionType::get(typeOf(type), makeArrayRef(argTypes), false);
  Function* function =
      Function::Create(ftype, GlobalValue::InternalLinkage, id.name, context.module.get());
  BasicBlock* bblock = BasicBlock::Create(myContext(), "entry", function, nullptr);

  context.pushBlock(bblock);

  Function::arg_iterator argsValues = function->arg_begin();
  Value* argumentValue = nullptr;

  for (it = arguments.begin(); it != arguments.end(); it++) {
    (**it).codeGen(context);

    argumentValue = &*argsValues++;
    argumentValue->setName((*it)->id.name);
    auto* inst = new StoreInst(argumentValue, context.locals()[(*it)->id.name], false, bblock);
    (void)inst;
  }

  block.codeGen(context);
  ReturnInst::Create(myContext(), context.getCurrentReturnValue(), bblock);

  context.popBlock();
  std::cout << "Creating function: " << id.name << endl;
  return function;
}
