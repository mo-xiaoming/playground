#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Compiler.h>

#include <iosfwd>

namespace dl {
struct Tokenizer {
  struct LLVM_NODISCARD Token {
  private:
    enum struct LLVM_NODISCARD Type : std::int8_t {
      UNKNOWN,
      FIN,
      PLUS,
      TIMES,
      MINUS,
      DIVIDE,
      NUMBER,
      KEYWORD,
      LCURLY,
      RCURLY,
      LPAREN,
      RPAREN,
      COMMA,
      PARROW,
      EQUAL,
      LID,
      UID,
    };

    LLVM_NODISCARD static llvm::StringRef toString(Type t);

  public:
    LLVM_NODISCARD friend bool operator==(Token const& lhs, Token const& rhs) {
      if (lhs.type_ != rhs.type_) {
        return false;
      }

      // following types can have different values
      static auto const ts =
          std::array{Type::KEYWORD, Type::NUMBER, Type::LID, Type::UID, Type::UNKNOWN};
      if (const auto* const it = std::find(ts.cbegin(), ts.cend(), lhs.type_); it != ts.cend()) {
        return lhs.value_ == rhs.value_;
      }

      return true;
    }

    LLVM_NODISCARD friend bool operator!=(Token const& lhs, Token const& rhs) {
      return !(lhs == rhs);
    }

    LLVM_NODISCARD llvm::StringRef value() const {
      return value_;
    }

    LLVM_NODISCARD std::string toString() const;

#define IS_TOKEN(x)                                                                                \
  LLVM_NODISCARD static bool is##x(Token t) {                                                      \
    return t == make##x();                                                                         \
  }
    static Token makeFin() {
      return {Type::FIN, ""};
    }
    IS_TOKEN(Fin)
    static Token makePlus() {
      return {Type::PLUS, "+"};
    }
    IS_TOKEN(Plus)
    static Token makeTimes() {
      return {Type::TIMES, "*"};
    }
    IS_TOKEN(Times)
    static Token makeMinus() {
      return {Type::MINUS, "-"};
    }
    IS_TOKEN(Minus)
    static Token makeDivide() {
      return {Type::DIVIDE, "/"};
    }
    IS_TOKEN(Divide)
    static Token makeLCurly() {
      return {Type::LCURLY, "{"};
    }
    IS_TOKEN(LCurly)
    static Token makeRCurly() {
      return {Type::RCURLY, "}"};
    }
    IS_TOKEN(RCurly)
    static Token makeLParen() {
      return {Type::LPAREN, "("};
    }
    IS_TOKEN(LParen)
    static Token makeRParen() {
      return {Type::RPAREN, ")"};
    }
    IS_TOKEN(RParen)
    static Token makeComma() {
      return {Type::COMMA, ","};
    }
    IS_TOKEN(Comma)
    static Token makePArrow() {
      return {Type::PARROW, "->"};
    }
    IS_TOKEN(PArrow)
    static Token makeEqual() {
      return {Type::EQUAL, "="};
    }
    IS_TOKEN(Equal)
    static Token makeKeywordData() {
      return {Type::KEYWORD, "data"};
    }
    IS_TOKEN(KeywordData)
    static Token makeKeywordDefn() {
      return {Type::KEYWORD, "defn"};
    }
    IS_TOKEN(KeywordDefn)
    static Token makeKeywordCase() {
      return {Type::KEYWORD, "case"};
    }
    IS_TOKEN(KeywordCase)
    static Token makeKeywordOf() {
      return {Type::KEYWORD, "of"};
    }
    IS_TOKEN(KeywordOf)
#undef IS_TOKEN
    static Token makeLID(llvm::StringRef s) {
      return {Type::LID, s};
    }
    static Token makeUID(llvm::StringRef s) {
      return {Type::UID, s};
    }
    static Token makeNumber(llvm::StringRef s) {
      return {Type::NUMBER, s};
    }
    LLVM_NODISCARD static bool isNumber(Token t) {
      return t.type_ == Type::NUMBER;
    }
    static Token makeUnknown(llvm::StringRef s) {
      return {Type::UNKNOWN, s};
    }

  private:
    Token(Type type, llvm::StringRef value) : type_(type), value_(value) {}

    Type type_ = Type::UNKNOWN;
    llvm::StringRef value_;
  };

  explicit Tokenizer(llvm::StringRef content) : content_(content) {}

  LLVM_NODISCARD Token next();

private:
  llvm::StringRef content_;
};
} // namespace dl

std::ostream& operator<<(std::ostream& os, dl::Tokenizer::Token const& t);
