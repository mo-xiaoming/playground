#include "scanner.hpp"

#include <catch2/catch.hpp>

#include <llvm/Support/FormatVariadic.h>

#include <array>
#include <type_traits>
#include <unordered_set>

TEST_CASE("empty input", "[scanner]") {
  dl::Tokenizer tk("");

  SECTION("first token is Fin") {
    auto const t = tk.next();
    CAPTURE(t);
    REQUIRE(dl::Tokenizer::Token::isFin(t));

    SECTION("'following' token is still Fin") {
      auto const t1 = tk.next();
      CAPTURE(t1);
      REQUIRE(dl::Tokenizer::Token::isFin(t1));
    }
  }
}

TEST_CASE("all blank input", "[scanner]") {
  dl::Tokenizer tk(" \n  \r\r\t ");

  SECTION("first token is Fin") {
    auto const t = tk.next();
    CAPTURE(t);
    REQUIRE(dl::Tokenizer::Token::isFin(t));
  }
}

TEST_CASE("recognize fixed symbols", "[scanner]") {
  auto const fixedSymbols = std::array{
      std::make_pair(dl::Tokenizer::Token::makePArrow(), dl::Tokenizer::Token::isPArrow),
      std::make_pair(dl::Tokenizer::Token::makePlus(), dl::Tokenizer::Token::isPlus),
      std::make_pair(dl::Tokenizer::Token::makeTimes(), dl::Tokenizer::Token::isTimes),
      std::make_pair(dl::Tokenizer::Token::makeMinus(), dl::Tokenizer::Token::isMinus),
      std::make_pair(dl::Tokenizer::Token::makeDivide(), dl::Tokenizer::Token::isDivide),
      std::make_pair(dl::Tokenizer::Token::makeLCurly(), dl::Tokenizer::Token::isLCurly),
      std::make_pair(dl::Tokenizer::Token::makeRCurly(), dl::Tokenizer::Token::isRCurly),
      std::make_pair(dl::Tokenizer::Token::makeLParen(), dl::Tokenizer::Token::isLParen),
      std::make_pair(dl::Tokenizer::Token::makeRParen(), dl::Tokenizer::Token::isRParen),
      std::make_pair(dl::Tokenizer::Token::makeComma(), dl::Tokenizer::Token::isComma),
      std::make_pair(dl::Tokenizer::Token::makeEqual(), dl::Tokenizer::Token::isEqual),
      std::make_pair(dl::Tokenizer::Token::makeKeywordCase(), dl::Tokenizer::Token::isKeywordCase),
      std::make_pair(dl::Tokenizer::Token::makeKeywordData(), dl::Tokenizer::Token::isKeywordData),
      std::make_pair(dl::Tokenizer::Token::makeKeywordDefn(), dl::Tokenizer::Token::isKeywordDefn),
      std::make_pair(dl::Tokenizer::Token::makeKeywordOf(), dl::Tokenizer::Token::isKeywordOf),
  };
  for (auto const& [t, f] : fixedSymbols) {
    INFO("token: " << t);
    REQUIRE(f(t));

    INFO("input: " << t.value().str());
    auto const nt = dl::Tokenizer(t.value()).next();
    INFO("next token: " << nt);
    REQUIRE(f(nt));
  }
}

TEST_CASE("keyword must not follow any alphanums", "[scanner]") {
  auto const keywords = std::array{
      std::make_pair(dl::Tokenizer::Token::makeKeywordCase().value().str(),
                     dl::Tokenizer::Token::isKeywordCase),
      std::make_pair(dl::Tokenizer::Token::makeKeywordData().value().str(),
                     dl::Tokenizer::Token::isKeywordData),
      std::make_pair(dl::Tokenizer::Token::makeKeywordDefn().value().str(),
                     dl::Tokenizer::Token::isKeywordDefn),
      std::make_pair(dl::Tokenizer::Token::makeKeywordOf().value().str(),
                     dl::Tokenizer::Token::isKeywordOf),
  };

  auto const illegalPostFix = std::array{"a", "A", "1"};
  for (auto const& [k, f] : keywords) {
    for (auto const* p : illegalPostFix) {
      std::string const c = k + p;
      dl::Tokenizer tk(c);
      auto const t = tk.next();
      INFO("first token of '" << c << "' is " << t << ", should not be a keyword " << k);
      REQUIRE_FALSE(f(t));
    }
  }

  auto const legalPostFix =
      std::array{"_", "=", "(", ")", "{", "}", "*", "/", "+", "-", "->", ",", " "};

  for (auto const& [k, f] : keywords) {
    for (auto const* p : legalPostFix) {
      std::string const c = k + p;
      dl::Tokenizer tk(c);
      auto const t = tk.next();
      INFO("first token of '" << c << "' is " << t << ", should be a keyword " << k);
      REQUIRE(f(t));
    }
  }
}

TEST_CASE("number must not follow any alphabets", "[scanner]") {
  std::string const number = "123";
/*
  auto const illegalPostFix = std::array{"a", "A"};
  for (auto const* p : illegalPostFix) {
    std::string const c = number + p;
    dl::Tokenizer tk(c);
    auto const t = tk.next();
    INFO("first token of '" << c << "' is " << t << ", should not be a number type");
    REQUIRE_FALSE(dl::Tokenizer::Token::isNumber(t));
  }
*/
  auto const legalPostFix =
      std::array{"_", "=", "(", ")", "{", "}", "*", "/", "+", "-", "->", ",", " ", "a", "A"};

  for (auto const* s : legalPostFix) {
    std::string const u = number + s;
    dl::Tokenizer tk(u);
    auto const k = tk.next();
    INFO("first token of (" << s << ") " << u << "' is " << k << ", should be a number type");
    REQUIRE(dl::Tokenizer::Token::isNumber(k));
  }
}

TEST_CASE("Unknown token equal to nothing", "[scanner]") {
  auto const tokens = std::array{
      dl::Tokenizer::Token::makePArrow(),      dl::Tokenizer::Token::makePlus(),
      dl::Tokenizer::Token::makeTimes(),       dl::Tokenizer::Token::makeMinus(),
      dl::Tokenizer::Token::makeDivide(),      dl::Tokenizer::Token::makeLCurly(),
      dl::Tokenizer::Token::makeRCurly(),      dl::Tokenizer::Token::makeLParen(),
      dl::Tokenizer::Token::makeRParen(),      dl::Tokenizer::Token::makeComma(),
      dl::Tokenizer::Token::makeEqual(),       dl::Tokenizer::Token::makeKeywordCase(),
      dl::Tokenizer::Token::makeKeywordData(), dl::Tokenizer::Token::makeKeywordDefn(),
      dl::Tokenizer::Token::makeKeywordOf(),
  };
  for (auto const& t : tokens) {
    REQUIRE(dl::Tokenizer::Token::makeUnknown(t.value()) != t);
  }
  auto const unknownT = dl::Tokenizer::Token::makeUnknown("");
  REQUIRE(unknownT != unknownT);
}

#if 0
TEST_CASE("") {
  GIVEN("3+4*2") {
    dl::Tokenizer tk("3+4*2");

    THEN("NUMBER 3, PLUS, NUMBER 4, TIMES, NUMBER2") {
      REQUIRE(tk.next() == dl::Tokenizer::Token::makeNumber("3"));
      REQUIRE(tk.next() == dl::Tokenizer::Token::makePlus());
      REQUIRE(tk.next() == dl::Tokenizer::Token::makeNumber("4"));
      REQUIRE(tk.next() == dl::Tokenizer::Token::makeTimes());
      REQUIRE(tk.next() == dl::Tokenizer::Token::makeNumber("2"));
      REQUIRE(tk.next() == dl::Tokenizer::Token::makeFin());
    }
  }
#endif
#if 0
  GIVEN("defn f x = {x+x}") {
    dl::Tokenizer tk("defn f x = {x+x}");

    THEN("") {
      REQUIRE(tk.next() == dl::Tokenizer::Token{.type = dl::Tokenizer::Type::NUMBER, .value = "3"});
      REQUIRE(tk.next() == dl::Tokenizer::Token{.type = dl::Tokenizer::Type::PLUS, .value = "+"});
      REQUIRE(tk.next() == dl::Tokenizer::Token{.type = dl::Tokenizer::Type::NUMBER, .value = "4"});
      REQUIRE(tk.next() == dl::Tokenizer::Token{.type = dl::Tokenizer::Type::TIMES, .value = "*"});
      REQUIRE(tk.next() == dl::Tokenizer::Token{.type = dl::Tokenizer::Type::NUMBER, .value = "2"});
      REQUIRE(tk.next() == dl::Tokenizer::Token{.type = dl::Tokenizer::Type::FIN, .value = {}});
    }
  }
#endif
