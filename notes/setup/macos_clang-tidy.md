brew install llvm
ln -s "$(brew --prefix llvm)/bin/clang-format" "/usr/local/bin/clang-format"
ln -s "$(brew --prefix llvm)/bin/clang-tidy" "/usr/local/bin/clang-tidy"
ln -s "$(brew --prefix llvm)/bin/clang-apply-replacements" "/usr/local/bin/clang-apply-replacements"
ln -s "$(brew --prefix llvm)/share/clang/run-clang-tidy.py" "/usr/local/bin/run-clang-tidy.py

find_program(CLANG_TIDY_BIN clang-tidy)
find_program(RUN_CLANG_TIDY_BIN /usr/local/Cellar/llvm/10.0.0_3/share/clang/run-clang-tidy.py)
list(APPEND RUN_CLANG_TIDY_BIN_ARGS
     -clang-tidy-binary ${CLANG_TIDY_BIN}
     -header-filter="\".*\\b(cpp-client/src)\""
     -checks=clan*,cert*,misc*,perf*,cppc*,read*,mode*,-cert-err58-cpp,-misc-noexcept-move-constructor,-cppcoreguidelines-*)

add_custom_target(tidy
                  COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS}
                  COMMENT "running clang tidy")
