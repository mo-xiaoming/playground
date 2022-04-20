%{
#include <stdio.h>

int yylex();
int yyerror(char* s);
%}

%token STRING NUM OTHER SIMICOLON PIC

%type <name> STRING
%type <number> NUM
%type <name> PIC

%union {
    char name[20];
    int number;
}

%%

prog:
    stmts
;

stmts:
     | stmt SEMICOLON stmts

stmt:
    STRING {
        printf("You entered a string - %s", $1);
    }
    | NUM {
        printf("The number you entered is - %d", $1);
    }
    | OTHER
;

%%

int yyerror(char* s) {
    printf("Syntax Error on line %s\n", s);
    return 0;
}

int main() {
    yyparse();
    return 0;
}
