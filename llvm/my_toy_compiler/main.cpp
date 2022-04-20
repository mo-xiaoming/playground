#include "codegen.h"
#include "node.h"
#include <iostream>

using namespace std;

extern int yyparse();
extern NBlock* programBlock;

void createCoreFunctions(CodeGenContext& context);

int main() {
	yyparse();
	cout << programBlock << endl;

	InitializeNativeTarget();
	InitializeNativeTargetAsmPrinter();
	InitializeNativeTargetAsmParser();
	CodeGenContext context;
	createCoreFunctions(context);
	context.generateCode(*programBlock);
	context.runCode();
}

