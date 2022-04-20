module Main (main) where

import qualified LearnYouGoodTest (tests)

import Test.HUnit
import System.Exit

miscTests :: Test
miscTests = "what if ..." ~: TestList [
  "number one" ~: True @? "this should be true"
  ]


main :: IO ()
main = do
  cnts <- runTestTT $ TestList [
    miscTests,
    LearnYouGoodTest.tests
    ]
  if errors cnts + failures cnts == 0
    then exitSuccess
    else exitFailure
