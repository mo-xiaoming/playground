module DataList (
  myIntersperse1,
  myIntercalate1,
  myTranspose1,
  myConcatMap1,
  myOr1,
  myAnd1,
  myAny1,
  myAll1,
  myIterate1,
  mySplitAt1,
  myTakeWhile1,
  myDropWhile1,
  mySpan1,
  myBreak1,
  myGroup1,
  myInits1,
  myTails1,
  myIsInfixOf1,
  myIsPrefixOf1,
  myIsSuffixOf1,
  myPartition1,
  myFind1,
  myElemIndex1,
  myElemIndices1,
  myFindIndex1,
  myFindIndices1,
  myZip31,
  myZipWith31,
  myLines1,
  myUnlines1,
  myUnwords1,
  myWords1,
  myNub1,
  myDelete1,
  myDifference1,
  myUnion1,
  myIntersect1,
  myInsert1,
  myGroupBy1,
) where

import HsPrelude

-- | intersperse
--
-- >>> myIntersperse1 '.' []
-- ""
--
-- >>> myIntersperse1 '.' "a"
-- "a"
--
-- >>> myIntersperse1 '.' "ab"
-- "a.b"
--
-- >>> myIntersperse1 '.' "abc"
-- "a.b.c"
myIntersperse1 :: a -> [a] -> [a]
myIntersperse1 _ [] = []
myIntersperse1 _ [x] = [x]
myIntersperse1 c (x:xs) = x:c:myIntersperse1 c xs

-- | concat
--
-- >>> myConcat1 ["abc", ""]
-- "abc"
--
-- >>> myConcat1 ["", "abc"]
-- "abc"
--
-- >>> myConcat1 ["abc"]
-- "abc"
--
-- >>> myConcat1 ["abc", "defg"]
-- "abcdefg"
myConcat1 :: [[a]] -> [a]
myConcat1 = myFoldr11 f
  where f :: [a] -> [a] -> [a]
        f [] acc     = acc
        f (y:ys) acc = y:f ys acc

-- | intercalate
--
-- >>> myIntercalate1 " " []
-- ""
--
-- >>> myIntercalate1 " " ["a"]
-- "a"
--
-- >>> myIntercalate1 " " ["a", "b"]
-- "a b"
--
-- >>> myIntercalate1 "12" ["ax", "by", "cz"]
-- "ax12by12cz"
myIntercalate1 :: [a] -> [[a]] -> [a]
myIntercalate1 _ [] = []
myIntercalate1 _ [[a]] = [a]
myIntercalate1 xs xss = myConcat1 (myIntersperse1 xs xss)

-- | transpose
--
-- >>> myTranspose1 []
-- []
--
-- >>> myTranspose1 [[]]
-- []
--
-- >>> myTranspose1 [[],[]]
-- []
--
-- >>> myTranspose1 [[],[1,2]]
-- [[1],[2]]
--
-- >>> myTranspose1 [[1,2],[]]
-- [[1],[2]]
--
-- >>> myTranspose1 [[1,2],[3,4]]
-- [[1,3],[2,4]]
--
-- >>> myTranspose1 [[1,2],[3,4,5]]
-- [[1,3],[2,4],[5]]
--
-- >>> myTranspose1 [[1,2,5,6],[3,4]]
-- [[1,3],[2,4],[5],[6]]
--
-- >>> myTranspose1 [[1,2,5,6],[3,4],[7]]
-- [[1,3,7],[2,4],[5],[6]]
--
-- >>> myTranspose1 [[1,2,5,6],[3,4],[7,8,9]]
-- [[1,3,7],[2,4,8],[5,9],[6]]
myTranspose1 :: [[a]] -> [[a]]
myTranspose1 [] = []
myTranspose1 ([]:xss) = myTranspose1 xss
myTranspose1 xss = ([h | (h:_) <- xss]):myTranspose1 [t |(_:t) <- xss]

-- | concatMap
--
-- >>> myConcatMap1 (myReplicate1 4) [1..3]
-- [1,1,1,1,2,2,2,2,3,3,3,3]
myConcatMap1 :: (a -> [a]) -> [a] -> [a]
myConcatMap1 _ [] = []
myConcatMap1 f a = myConcat1 (map f a)

-- | or
--
-- >>> myOr1 []
-- False
--
-- >>> myOr1 [True]
-- True
--
-- >>> myOr1 [False]
-- False
--
-- >>> myOr1 [True, False]
-- True
--
-- >>> myOr1 [False, True]
-- True
--
-- >>> myOr1 [False, False]
-- False
myOr1 :: [Bool] -> Bool
myOr1 [] = False
myOr1 (x:xs) | x = True
             | otherwise = myOr1 xs

-- | and
--
-- >>> myAnd1 []
-- True
--
-- >>> myAnd1 [True]
-- True
--
-- >>> myAnd1 [False]
-- False
--
-- >>> myAnd1 [True, False]
-- False
--
-- >>> myAnd1 [False, True]
-- False
--
-- >>> myAnd1 [False, False]
-- False
--
-- >>> myAnd1 [True, True]
-- True
myAnd1 :: [Bool] -> Bool
myAnd1 [] = True
myAnd1 (x:xs) | not x = False
              | otherwise = myAnd1 xs

-- | any
--
-- >>> myAny1 (==4) [2,3,5,6,1,4]
-- True
--
-- >>> myAny1 (`elem` ['A'..'Z']) "HEYGUYSwhatsup"
-- True
myAny1 :: (a -> Bool) -> [a] -> Bool
myAny1 f a = myOr1 (myMap1 f a)

-- | all
--
-- >>> myAll1 (>4) [6,9,10]
-- True
--
-- >>> all (`elem` ['A'..'Z']) "HEYGUYSwhatsup"
-- False
myAll1 :: (a -> Bool) -> [a] -> Bool
myAll1 f a = myAnd1 (myMap1 f a)

-- | iterate
--
-- >>> myTake1 10 $ myIterate1 (*2) 1
-- [1,2,4,8,16,32,64,128,256,512]
--
-- >>> myTake1 3 $ myIterate1 (++ "haha") "haha"
-- ["haha","hahahaha","hahahahahaha"]
myIterate1 :: (a -> a) -> a -> [a]
myIterate1 f a = a:myIterate1 f (f a)

-- | splitAt
--
-- >>> mySplitAt1 3 "heyman"
-- ("hey","man")
--
-- >>> mySplitAt1 100 "heyman"
-- ("heyman","")
--
-- >>> mySplitAt1 (-3) "heyman"
-- ("","heyman")
mySplitAt1 :: Int -> [a] -> ([a], [a])
mySplitAt1 _ [] = ([],[])
mySplitAt1 i a | i <= 0 = ([],a)
mySplitAt1 i (x:xs) = (x:fst n, snd n)
  where n = mySplitAt1 (i-1) xs
{-# ANN mySplitAt1 "HLint: Use first" #-}

-- | takeWhile
--
-- >>> myTakeWhile1 (>3) [6,5,4,3,2,1,2,3,4,5,4,3,2,1]
-- [6,5,4]
--
-- >>> myTakeWhile1 (/=' ') "This is a sentence"
-- "This"
--
-- >>> myTakeWhile1 (<3) [1..]
-- [1,2]
myTakeWhile1 :: (a -> Bool) -> [a] -> [a]
myTakeWhile1 _ [] = []
myTakeWhile1 f (x:xs) = if f x then x:myTakeWhile1 f xs else []

-- |  dropWhile
--
-- >>> myDropWhile1 (/=' ') "This is a sentence"
-- " is a sentence"
--
-- >>> myDropWhile1 (<3) [1,2,2,2,3,4,5,4,3,2,1]
-- [3,4,5,4,3,2,1]
myDropWhile1 :: (a -> Bool) -> [a] -> [a]
myDropWhile1 _ [] = []
myDropWhile1 f a@(x:xs) = if f x then myDropWhile1 f xs else a

-- | span
--
-- >>> mySpan1 (/=4) [1,2,3,4,5,6,7]
-- ([1,2,3],[4,5,6,7])
mySpan1 :: (a -> Bool) -> [a] -> ([a],[a])
mySpan1 _ [] = ([],[])
mySpan1 f a@(x:xs) | f x = let (ys,zs) = mySpan1 f xs in (x:ys,zs)
                   | otherwise = ([],a)

-- | break
--
-- >>> myBreak1 (==4) [1,2,3,4,5,6,7]
-- ([1,2,3],[4,5,6,7])
myBreak1 :: (a -> Bool) -> [a] -> ([a],[a])
myBreak1 f = mySpan1 (not . f)

-- | groupBy
--
-- >>> let values = [-4.3, -2.4, -1.2, 0.4, 2.3, 5.9, 10.5, 29.1, 5.3, -2.4, -14.5, 2.9, 2.3]
-- >>> myGroupBy1 (\x y -> (x > 0) == (y > 0)) values
-- [[-4.3,-2.4,-1.2],[0.4,2.3,5.9,10.5,29.1,5.3],[-2.4,-14.5],[2.9,2.3]]
myGroupBy1 :: (a -> a -> Bool) -> [a] -> [[a]]
myGroupBy1 _ [] = []
myGroupBy1 _ [x] = [[x]]
myGroupBy1 f a = let n = mySpan1 (f $ head a) a in fst n:myGroupBy1 f (snd n)
-- | group
--
-- >>> myGroup1 [1,1,1,1,2,2,2,2,3,3,2,2,2,5,6,7]
-- [[1,1,1,1],[2,2,2,2],[3,3],[2,2,2],[5],[6],[7]]
--
-- >>> myMap1 (\l@(x:xs) -> (x,myLength1 l)) . myGroup1 . myQSort2 $ [1,1,1,1,2,2,2,2,3,3,2,2,2,5,6,7]
-- [(1,4),(2,7),(3,2),(5,1),(6,1),(7,1)]
myGroup1 :: Eq a => [a] -> [[a]]
myGroup1 a = myGroupBy1 (==) a

-- | inits
--
-- >>> myInits1 []
-- [[]]
--
-- >>> myInits1 [1]
-- [[],[1]]
--
-- >>> myInits1 "w00t"
-- ["","w","w0","w00","w00t"]
myInits1 :: [a] -> [[a]]
myInits1 a = myFoldr1 (\_ acc -> (myInit1 $ head acc):acc) [a] a

-- | tails
--
-- >>> myTails1 []
-- [[]]
--
-- >>> myTails1 [1]
-- [[1],[]]
--
-- >>> myTails1 "w00t"
-- ["w00t","00t","0t","t",""]
myTails1 :: [a] -> [[a]]
myTails1 a = myFoldr1 (\x acc -> (x:head acc):acc) [[]] a

-- | isInfixOf
--
-- >>> myIsInfixOf1 "cat" "im a cat burglar"
-- True
--
-- >>> myIsInfixOf1 "Cat" "im a cat burglar"
-- False
--
-- >>> myIsInfixOf1 "cats" "im a cat burglar"
-- False
myIsInfixOf1 :: Eq a => [a] -> [a] -> Bool
myIsInfixOf1 s a = let n = myLength1 s in myAny1 (\x -> myTake1 n x == s) (myTails1 a)

-- | isPrefixOf
--
-- >>> myIsPrefixOf1 "hey" "hey there!"
-- True
--
-- >>> myIsPrefixOf1 "hey" "oh hey there!"
-- False
myIsPrefixOf1 :: Eq a => [a] -> [a] -> Bool
myIsPrefixOf1 s a = s == myTake1 (myLength1 s) a

-- | isSuffixOf
--
-- >>> myIsSuffixOf1 "there!" "oh hey there!"
-- True
--
-- >>> myIsSuffixOf1 "there!" "oh hey there"
-- False
myIsSuffixOf1 :: Eq a => [a] -> [a] -> Bool
myIsSuffixOf1 s a = myIsPrefixOf1 (myReverse1 s) (myReverse2 a)

-- | partition
--
-- >>> myPartition1 (>3) []
-- ([],[])
--
-- >>> myPartition1 (>3) [1,3,5,6,3,2,1,0,3,7]
-- ([5,6,7],[1,3,3,2,1,0,3])
--
-- >>> myPartition1 (`myElem1` ['A'..'Z']) "BOBsidneyMORGANeddy"
-- ("BOBMORGAN","sidneyeddy")
myPartition1 :: (a -> Bool) -> [a] -> ([a],[a])
myPartition1 f a = myFoldr1 (\x acc -> if f x then (x:fst acc,snd acc) else (fst acc,x:snd acc)) ([],[]) a

-- | find
--
-- >>> myFind1 (>4) [1,2,3,4,5,6]
-- Just 5
-- >>> myFind1 (>9) [1,2,3,4,5,6]
-- Nothing
myFind1 :: (a -> Bool) -> [a] -> Maybe a
myFind1 _ [] = Nothing
myFind1 f (x:xs) | f x = Just x
                 | otherwise = myFind1 f xs

-- | findIndices
--
-- >>> myFindIndices1 (`elem` ['A'..'Z']) "Where Are The Caps?"
-- [0,6,10,14]
myFindIndices1 :: Eq a => (a -> Bool) -> [a] -> [Int]
myFindIndices1 p a = [i | (x,i) <- zip a [0..], p x]

-- | findIndex
--
-- >>> myFindIndex1 (==4) [5,3,2,1,6,4]
-- Just 5
-- >>> myFindIndex1 (==7) [5,3,2,1,6,4]
-- Nothing
myFindIndex1 :: Eq a => (a -> Bool) -> [a] -> Maybe Int
myFindIndex1 p a = let n = myFindIndices1 p a in if null n then Nothing else Just $ head n

-- | elemIndex
--
-- >>> 4 `myElemIndex1` [1,2,3,4,5,6]
-- Just 3
--
-- >>> 10 `myElemIndex1` [1,2,3,4,5,6]
-- Nothing
myElemIndex1 :: Eq a => a -> [a] -> Maybe Int
myElemIndex1 e = myFindIndex1 (==e)

-- | elemIndices
--
-- >>> ' ' `myElemIndices1` "Where are the spaces?"
-- [5,9,13]
--
-- >>> 'A' `myElemIndices1` "Where are the spaces?"
-- []
myElemIndices1 :: Eq a => a -> [a] -> [Int]
myElemIndices1 e = myFindIndices1 (==e)

-- | zip3
--
-- >>> myZip31 [2,3,3] [2,2,2] [5,5,3]
-- [(2,2,5),(3,2,5),(3,2,3)]
myZip31 :: [a] -> [b] -> [c] -> [(a,b,c)]
myZip31 [] _ _ = []
myZip31 _ [] _ = []
myZip31 _ _ [] = []
myZip31 (x:xs) (y:ys) (z:zs) = (x,y,z):myZip31 xs ys zs

-- | zipWith3
--
-- >>> myZipWith31 (\x y z -> x + y + z) [1,2,3] [4,5,2,2] [2,2,3]
-- [7,9,8]
myZipWith31 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
myZipWith31 _ [] _ _ = []
myZipWith31 _ _ [] _ = []
myZipWith31 _ _ _ [] = []
myZipWith31 f (x:xs) (y:ys) (z:zs) = f x y z :myZipWith31 f xs ys zs

mySplit1 :: (a -> Bool) -> [a] -> [[a]]
mySplit1 _ [] = []
mySplit1 p a = let (h,t) = myBreak1 p a in h:(if null t then [] else (mySplit1 p (myTail1 t)))

-- | lines
--
-- >>> myLines1 []
-- []
--
-- >>> myLines1 "abc"
-- ["abc"]
--
-- >>> myLines1 "\n"
-- [""]
--
-- >>> myLines1 "\na"
-- ["","a"]
--
-- >>> myLines1 "a\n"
-- ["a"]
--
-- >>> myLines1 "first line\nsecond line\nthird line"
-- ["first line","second line","third line"]
myLines1 :: [Char] -> [[Char]]
myLines1 a = mySplit1 (=='\n') a

-- | unlines
--
-- >>> myUnlines1 [""]
-- "\n"
--
-- >>> myUnlines1 ["a"]
-- "a\n"
--
-- >>> myUnlines1 ["first line", "second line", "third line"]
-- "first line\nsecond line\nthird line\n"
myUnlines1 :: [[Char]] -> [Char]
myUnlines1 xss = (myIntercalate1 "\n" xss) ++ "\n"

-- | unwords
--
-- >>> myUnwords1 ["hey","there","mate"]
-- "hey there mate"
myUnwords1 :: [[Char]] -> [Char]
myUnwords1 = myIntercalate1 " "

-- | words
--
-- >>> myWords1 []
-- []
--
-- >>> myWords1 "   "
-- []
--
-- >>> myWords1 " abc   "
-- ["abc"]
--
-- >>> myWords1 "hey these are the words in this sentence"
-- ["hey","these","are","the","words","in","this","sentence"]
--
-- >>> myWords1 "hey these           are    the words in this\nsentence"
-- ["hey","these","are","the","words","in","this","sentence"]
myWords1 :: [Char] -> [[Char]]
myWords1 a = myFilter1 (/="") (mySplit1 f a)
  where f x = myElem1 x [' ', '\n']

-- |  nub
--
-- >>> myNub1 [1,2,3,4,3,2,1,2,3,4,3,2,1]
-- [1,2,3,4]
--
-- >>> myNub1 "Lots of words and stuff"
-- "Lots fwrdanu"
myNub1 :: Eq a => [a] -> [a]
myNub1 = myReverse2 . myFoldl1 (\acc x -> if myElem1 x acc then acc else x:acc) []

-- | delete
--
-- >>> myDelete1 'h' "hey there ghang!"
-- "ey there ghang!"
--
-- >>> myDelete1 'h' . myDelete1 'h' $ "hey there ghang!"
-- "ey tere ghang!"
--
-- >>> myDelete1 'h' . myDelete1 'h' . myDelete1 'h' $ "hey there ghang!"
-- "ey tere gang!"
myDelete1 :: Eq a => a -> [a] -> [a]
myDelete1 _ [] = []
myDelete1 c (x:xs) | c == x = xs
                   | otherwise = x:myDelete1 c xs

-- | \\
--
-- >>> myDifference1 "abab" "aa"
-- "bb"
--
-- >>> myDifference1 [1..10] [2,5,9]
-- [1,3,4,6,7,8,10]
--
-- >>> myDifference1 "Im a big baby" "big"
-- "Im a  baby"
myDifference1 :: Eq a => [a] -> [a] -> [a]
myDifference1 xs ys = myFoldl1 (\acc x -> myDelete1 x acc) xs ys

-- | union
--
-- >>> myUnion1 "hey man" "man what's up"
-- "hey manwt'sup"
--
-- >>> myUnion1 [1..7] [5..10]
-- [1,2,3,4,5,6,7,8,9,10]
myUnion1 :: Eq a => [a] -> [a] -> [a]
myUnion1 a [] = a
myUnion1 a (x:xs) = if myElem1 x a then myUnion1 a xs else myUnion1 (a ++ [x]) xs

-- | intersect
--
-- >>> myIntersect1 [1..7] [5..10]
-- [5,6,7]
myIntersect1 :: Eq a => [a] -> [a] -> [a]
myIntersect1 [] _ = []
myIntersect1 _ [] = []
myIntersect1 a b = [i | i<- a, myElem1 i b]

-- | insert
--
-- >>> myInsert1 4 [3,5,1,2,8,2]
-- [3,4,5,1,2,8,2]
--
-- >>> myInsert1 4 [1,3,4,4,1]
-- [1,3,4,4,4,1]
--
-- myInsert1 4 [1,2,3,5,6,7]
-- [1,2,3,4,5,6,7]
--
-- >>> myInsert1 'g' $ ['a'..'f'] ++ ['h'..'z']
-- "abcdefghijklmnopqrstuvwxyz"
--
-- >>> myInsert1 3 [1,2,4,3,2,1]
-- [1,2,3,4,3,2,1]
myInsert1 :: Ord a => a -> [a] -> [a]
myInsert1 i [] = [i]
myInsert1 i a@(x:xs) | i >  x = x:myInsert1 i xs
                     | otherwise = i:a

