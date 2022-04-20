%{
#include <stdio.h>
#include <string.h>
#include <test.tab.h>
void showError();
%}

number ([0-9])+
alpha ([a-zA-Z])+

%%

"{alpha}" {sscanf(yytext, "%s", yylval.name); return STRING;}
{number} {yylval.number = atoi(yytext); return NUM;}
";" {return SEMICOLON;}
. {showError(); return OTHER;}

%%

void showError() {
    printf("Other input");
}
