#include "scanner.hpp"

#include <llvm/ADT/Optional.h>
#include <spdlog/fmt/fmt.h>

namespace {
LLVM_NODISCARD bool charBelongsTo(char c, const llvm::StringRef sset) {
  return sset.contains(c);
}

LLVM_NODISCARD bool isNum(char c) {
  return charBelongsTo(c, "0123456789");
}

LLVM_NODISCARD bool isLowerAlpha(char c) {
  return charBelongsTo(c, "abcdefghijklmnopqrstuvwxyz");
}

LLVM_NODISCARD bool isUpperAlpha(char c) {
  return charBelongsTo(c, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

LLVM_NODISCARD bool isAlpha(char c) {
  return isLowerAlpha(c) || isUpperAlpha(c);
}

LLVM_NODISCARD bool isAlphaNum(char c) {
  return isAlpha(c) || isNum(c);
}

void skipBlanks(llvm::StringRef& content) {
  content = content.drop_while([](char c) { return charBelongsTo(c, " \t\r\n"); });
}

LLVM_NODISCARD bool reachFIN(llvm::StringRef content) {
  return content.empty();
}

LLVM_NODISCARD llvm::Optional<dl::Tokenizer::Token> scanFixedSymbol(llvm::StringRef& content) {
  // order is important, longest to be first to be greedy?
  // e.g. `->` must appear earlier than `-`
  static auto const fixedSymbols = std::array{
      dl::Tokenizer::Token::makePArrow(), dl::Tokenizer::Token::makePlus(),
      dl::Tokenizer::Token::makeTimes(),  dl::Tokenizer::Token::makeMinus(),
      dl::Tokenizer::Token::makeDivide(), dl::Tokenizer::Token::makeLCurly(),
      dl::Tokenizer::Token::makeRCurly(), dl::Tokenizer::Token::makeLParen(),
      dl::Tokenizer::Token::makeRParen(), dl::Tokenizer::Token::makeComma(),
      dl::Tokenizer::Token::makeEqual(),
  };
  for (auto const& s : fixedSymbols) {
    if (content.startswith(s.value())) {
      content = content.drop_front(s.value().size());
      return s;
    }
  }
  return {};
}

LLVM_NODISCARD llvm::Optional<dl::Tokenizer::Token> scanKeyword(llvm::StringRef& content) {
  // order is important, longest to be first to be greedy?
  static auto const keywords = std::array{
      dl::Tokenizer::Token::makeKeywordData(),
      dl::Tokenizer::Token::makeKeywordDefn(),
      dl::Tokenizer::Token::makeKeywordCase(),
      dl::Tokenizer::Token::makeKeywordOf(),
  };
  for (auto const& k : keywords) {
    if (!content.startswith(k.value())) {
      continue;
    }
    // not followed by alphnum or it is the last token
    if ((content.size() > k.value().size() && !isAlphaNum(content[k.value().size()])) ||
        (content.size() == k.value().size())) {
      content = content.drop_front(k.value().size());
      return k;
    }
  }
  return {};
}

LLVM_NODISCARD llvm::Optional<dl::Tokenizer::Token> scanNumber(llvm::StringRef& content) {
  auto const new_content = content.drop_while(isNum);
  if (new_content.empty() || !isAlpha(new_content[0])) {
    auto const t =
        dl::Tokenizer::Token::makeNumber({content.data(), content.size() - new_content.size()});
    content = new_content;
    return t;
  }
  return {};
}

LLVM_NODISCARD llvm::Optional<dl::Tokenizer::Token> scanID(llvm::StringRef& content) {
  auto const new_content = content.drop_while(isAlpha);
  if (new_content != content) {
    dl::Tokenizer::Token const t = [=]() {
      llvm::StringRef const s(content.data(), content.size() - new_content.size());
      if (isLowerAlpha(content[0])) {
        return dl::Tokenizer::Token::makeLID(s);
      }
      return dl::Tokenizer::Token::makeUID(s);
    }();
    content = new_content;
    return t;
  }
  return {};
}
} // namespace

namespace dl {
llvm::StringRef dl::Tokenizer::Token::toString(dl::Tokenizer::Token::Type t) {
  struct S {
    Type type = Type::UNKNOWN;
    llvm::StringRef str;
  };
  static std::array const a = {
      S{.type = Type::UNKNOWN, .str = "UNKNOWN"}, S{.type = Type::FIN, .str = "FIN"},
      S{.type = Type::PLUS, .str = "PLUS"},       S{.type = Type::TIMES, .str = "TIMES"},
      S{.type = Type::MINUS, .str = "MINUS"},     S{.type = Type::DIVIDE, .str = "DIVIDE"},
      S{.type = Type::NUMBER, .str = "NUMBER"},   S{.type = Type::KEYWORD, .str = "KEYWORD"},
      S{.type = Type::LCURLY, .str = "LCURLY"},   S{.type = Type::RCURLY, .str = "RCURLY"},
      S{.type = Type::LPAREN, .str = "LPAREN"},   S{.type = Type::RCURLY, .str = "RPAREN"},
      S{.type = Type::COMMA, .str = "COMMA"},     S{.type = Type::PARROW, .str = "PARROW"},
      S{.type = Type::EQUAL, .str = "EQUAL"},     S{.type = Type::LID, .str = "LID"},
      S{.type = Type::UID, .str = "UID"},
  };
  if (const auto* const it =
          std::find_if(a.cbegin(), a.cend(), [t](S const& s) { return s.type == t; });
      it != a.cend()) {
    return it->str;
  }
  return "?";
}

std::string Tokenizer::Token::toString() const {
  return fmt::format(FMT_STRING("Token{{.type={0}, .value={1}}}"), toString(type_), value_);
}

LLVM_NODISCARD Tokenizer::Token Tokenizer::next() {
  skipBlanks(content_);

  if (reachFIN(content_)) {
    return Token::makeFin();
  }

  if (auto const t = scanFixedSymbol(content_)) {
    return *t;
  }

  if (auto const t = scanKeyword(content_)) {
    return *t;
  }

  if (auto const t = scanNumber(content_)) {
    return *t;
  }

  if (auto const t = scanID(content_)) {
    return *t;
  }

  return Token::makeUnknown({content_.data(), 1});
}
} // namespace dl

std::ostream& operator<<(std::ostream& os, dl::Tokenizer::Token const& t) {
  return os << t.toString();
}
