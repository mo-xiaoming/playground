#pragma once

/*
 * three := 3
 * seven := one + two + three
 * seven * seven
 */

#include <string>
#include <map>
#include "parser.tab.hh"

#define YY_DECL \
  yy::parser::symbol_type yylex(driver& drv)

YY_DECL;

class driver {
  public:
    driver();

    std::map<std::string, int> variables;

    int result;

    std::string file;

    int parse(const std::string& f);
    bool trace_parsing = false;

    void scan_begin();
    void scan_end();
    bool trace_scanning = false;

    yy::location location;
};

