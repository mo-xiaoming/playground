// http://aquamentus.com/flex_bison.html

%{
#include <cstdio>
#include <iostream>
using namespace std;

extern int yylex();
extern int yyparse();
extern FILE* yyin;
extern int line_num;

void yyerror(const char* s);
%}

%union {
    int ival;
    float fval;
    char* sval;
}

// constant-string
%token SNAZZLE TYPE
%token END ENDL

// terminal-symbol
%token <ival> INT
%token <fval> FLOAT
%token <sval> STRING

%%

// the first rtule defined is the highest-level rule, which in our
// case is just the concept of a whole "snazzle file"
snazzle:
    header template body_section footer {
        cout << "done with a snazzle file!" << endl;
    }
    ;
header:
    SNAZZLE FLOAT ENDLS {
        cout << "reading a snazzle file version " << $2 << endl;
    }
    ;
template:
    typelines
    ;
typelines:
    typelines typeline
    | typeline
    ;
typeline:
    TYPE STRING ENDLS {
        cout << "new defined snazzle type: " << $2 << endl;
    }
    ;
body_section:
    body_lines
    ;
body_lines:
    body_lines body_line
    | body_line
    ;
body_line:
    INT INT INT INT STRING ENDL {
        cout << "new snazzle: " << $1 << $2 << $3 << $4 << $5 << endl;
        free($5);
    }
    ;
footer:
    END ENDLS
    ;
ENDLS:
    ENDLS ENDL
    | ENDL
    ;
%%

int main() {
    FILE* myfile = fopen("in.snazzle", "r");
    if (myfile == nullptr) {
        cout << "I can't open a.snazzle.file!" << endl;
        return -1;
    }
    // set lex to read from it instead of defaulting to STDIN
    yyin = myfile;

    yyparse();
}

void yyerror(const char* s) {
    cout << "EEK, parse error on line " << line_num << "! Message: " << s << endl;
    exit(-1);
}
