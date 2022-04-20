import Test.DocTest

main :: IO ()
main = doctest ["-isrc", "src/HsPrelude.hs", "src/DataList.hs"]
