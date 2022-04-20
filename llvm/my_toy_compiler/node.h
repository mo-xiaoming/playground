#include <iostream>
#include <llvm/IR/Value.h>
#include <vector>

class CodeGenContext;
class NStatement;
class NExpression;
class NVariableDeclaration;

using StatementList = std::vector<NStatement *>;
using ExpressionList = std::vector<NExpression *>;
using VariableList = std::vector<NVariableDeclaration *>;

class Node {
public:
  virtual ~Node() = default;
  virtual llvm::Value* codeGen(CodeGenContext&  /*context*/) { return nullptr; }
};

class NExpression : public Node {};

class NStatement : public Node {};

class NInteger : public NExpression {
public:
  long long value;
  NInteger(long long value) : value(value) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NDouble : public NExpression {
public:
  double value;
  NDouble(double value) : value(value) {}
  virtual llvm::Value* codeGen(CodeGenContext& context);
};

class NIdentifier : public NExpression {
public:
  std::string name;
  NIdentifier(const std::string& name) : name(name) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NMethodCall : public NExpression {
public:
  const NIdentifier& id;
  ExpressionList arguments;
  NMethodCall(const NIdentifier& id, ExpressionList& arguments) : id(id), arguments(arguments) {}
  explicit NMethodCall(const NIdentifier& id) : id(id) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NBinaryOperator : public NExpression {
public:
  int op;
  NExpression& lhs;
  NExpression& rhs;
  NBinaryOperator(NExpression& lhs, int op, NExpression& rhs) : op(op), lhs(lhs), rhs(rhs) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NAssignment : public NExpression {
public:
  NIdentifier& lhs;
  NExpression& rhs;
  NAssignment(NIdentifier& lhs, NExpression& rhs) : lhs(lhs), rhs(rhs) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NBlock : public NExpression {
public:
  StatementList statements;
  NBlock() {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NExpressionStatement : public NStatement {
public:
  NExpression& expression;
  NExpressionStatement(NExpression& expression) : expression(expression) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NReturnStatement : public NStatement {
public:
  NExpression& expression;
  NReturnStatement(NExpression& expression) : expression(expression) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NVariableDeclaration : public NStatement {
public:
  const NIdentifier& type;
  NIdentifier& id;
  NExpression* assignmentExpr;
  NVariableDeclaration(const NIdentifier& type, NIdentifier& id) : type(type), id(id) {
    assignmentExpr = nullptr;
  }
  NVariableDeclaration(const NIdentifier& type, NIdentifier& id, NExpression* assignmentExpr)
      : type(type), id(id), assignmentExpr(assignmentExpr) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NExternDeclaration : public NStatement {
public:
  const NIdentifier& type;
  const NIdentifier& id;
  VariableList arguments;
  NExternDeclaration(const NIdentifier& type, const NIdentifier& id, VariableList  arguments)
      : type(type), id(id), arguments(std::move(arguments)) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};

class NFunctionDeclaration : public NStatement {
public:
  const NIdentifier& type;
  const NIdentifier& id;
  VariableList arguments;
  NBlock& block;
  NFunctionDeclaration(const NIdentifier& type, const NIdentifier& id,
                       const VariableList& arguments, NBlock& block)
      : type(type), id(id), arguments(arguments), block(block) {}
   llvm::Value* codeGen(CodeGenContext& context) override;
};
