{-
import Data.List
import System.IO

calcBmis :: (RealFloat a) => [(a, a)] -> [a]
calcBmis xs = [bmi w h | (w, h) <- xs]
    where bmi weight height = weight / height ^ 2
-}

{-
main = interact lineCount
    where lineCount input = show (length (lines input)) ++ "\n"
-}

{-
main = interact wordCount
    where wordCount input = show (length (words input)) ++ "\n"
-}

main = interact charCount
    where charCount inp = show (length inp) ++ "\n"
