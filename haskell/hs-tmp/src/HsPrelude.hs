module HsPrelude (
  myOdd1,
  myEven1,
  myMin1,
  myMax1,
  myHead1,
  myTail1,
  myLast1,
  myInit1,
  myLength1,
  myNull1,
  myReverse1,
  myReverse2,
  myTake1,
  myDrop1,
  myMinimum1,
  myMinimum2,
  myMaximum1,
  myMaximum2,
  mySum1,
  mySum2,
  myProduct1,
  myProduct2,
  myElem1,
  myNotElem1,
  myCycle1,
  myRepeat1,
  myReplicate1,
  myReplicate2,
  myFst1,
  mySnd1,
  myZip1,
  myQSort1,
  myQSort2,
  myZipWith1,
  myFlip1,
  myMap1,
  myMap2,
  myFilter1,
  myFoldl1,
  myFoldl11,
  myFoldr1,
  myFoldr11,
  myScanl1,
  myScanl11,
  myScanr1,
  myScanr11
) where

errorEmptyList :: a
errorEmptyList = error "empty list"

-- | odd
--
-- >>> myOdd1 3 == odd 3
-- True
--
-- >>> myOdd1 4 == odd 4
-- True
--
-- >>> myOdd1 0 == odd 0
-- True
myOdd1 :: Int -> Bool
myOdd1 a = mod a 2 == 1
{-# ANN myOdd1 "HLint: ignore Use odd" #-}

-- | even
--
-- >>> myEven1 3 == even 3
-- True
--
-- >>> myEven1 4 == even 4
-- True
--
-- >>> myEven1 0 == even 0
-- True
myEven1 :: Int -> Bool
myEven1 a = mod a 2 == 0
{-# ANN myEven1 "HLint: ignore Use even" #-}

-- | min
--
-- >>> myMin1 1 2 == min 1 2
-- True
--
-- >>> myMin1 2.0 1.0 == min 2.0 1.0
-- True
myMin1 :: (Num a, Ord a) => a -> a -> a
myMin1 a b = if a > b then b else a

-- | max
--
-- >>> myMax1 1 2 == max 1 2
-- True
--
-- >>> myMax1 2.0 1.0 == max 2.0 1.0
-- True
myMax1 :: (Ord a) => a -> a -> a
myMax1 a b = if a > b then a else b

-- | head
--
-- >>> head []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myHead1 []
-- *** Exception: empty list
-- ...
--
-- >>> myHead1 [1] == head [1]
-- True
--
-- >>> myHead1 [1..]
-- 1
--
-- >>> myHead1 [1, 2, 3]
-- 1
myHead1 :: [a] -> a
myHead1 [] = errorEmptyList
myHead1 (x:_) = x

-- | tail
--
-- >>> tail []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myTail1 []
-- *** Exception: empty list
-- ...
--
-- >>> myTail1 [1] == tail [1]
-- True
--
-- >>> myTail1 [1, 2] == tail [1, 2]
-- True
--
-- >>> myTail1 [1, 2, 3] == tail [1, 2, 3]
-- True
myTail1 :: [a] -> [a]
myTail1 [] = errorEmptyList
myTail1 (_:xs) = xs

-- | last
--
-- >>> last []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myLast1 []
-- *** Exception: empty list
-- ...
--
-- >>> myLast1 [1] == last [1]
-- True
--
-- >>> myLast1 [1, 2, 3] == last [1, 2, 3]
-- True
myLast1 :: [a] -> a
myLast1 [] = errorEmptyList
myLast1 [x] = x
myLast1 (_:xs) = myLast1 xs

-- | init
--
-- >>> init []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myInit1 []
-- *** Exception: empty list
-- ...
--
-- >>> myInit1 [1] == init [1]
-- True
--
-- >>> myInit1 [1, 2, 3] == init [1, 2, 3]
-- True
myInit1 :: [a] -> [a]
myInit1 [] = errorEmptyList
myInit1 [_] = []
myInit1 (x:xs) = x:myInit1 xs

-- | length
--
-- >>> myLength1 [] == length []
-- True
--
-- >>> myLength1 [1, 2, 3] == length [1, 2, 3]
-- True
myLength1 :: [a] -> Int
myLength1 [] = 0
myLength1 (_:xs) = 1 + length xs

-- | null
--
-- >>> myNull1 [] == null []
-- True
--
-- >>> myNull1 [1] == null [1]
-- True
--
-- >>> myNull1 [1..] == null [1..]
-- True
myNull1 :: [a] -> Bool
myNull1 [] = True
myNull1 _  = False

-- | reverse
--
-- >>> myReverse1 [] == reverse []
-- True
--
-- >>> myReverse1 [1] == reverse [1]
-- True
--
-- >>> myReverse1 [1, 2, 3] == reverse [1, 2, 3]
-- True
--
-- >>> myReverse2 [] == reverse []
-- True
--
-- >>> myReverse2 [1] == reverse [1]
-- True
--
-- >>> myReverse2 [1, 2, 3] == reverse [1, 2, 3]
-- True
myReverse1 :: [a] -> [a]
myReverse1 [] = []
myReverse1 [x] = [x]
myReverse1 (x:xs) = myReverse1 xs ++ [x]
myReverse2 :: [a] -> [a]
myReverse2 = myFoldl1 (flip (:)) []

-- | take
--
-- >>> myTake1 10 [] == take 10 []
-- True
--
-- >>> myTake1 0 [] == take 0 []
-- True
--
-- >>> myTake1 0 [1, 2, 3] == take 0 [1, 2, 3]
-- True
--
-- >>> myTake1 2 [1, 2] == take 2 [1, 2]
-- True
--
-- >>> myTake1 3 [1, 2] == take 3 [1, 2]
-- True
--
-- >>> myTake1 3 [4..] == take 3 [4..]
-- True
--
-- >>> myTake1 (-1) [4..] == take (-1) [4..]
-- True
myTake1 :: Int -> [a] -> [a]
myTake1 _ [] = []
myTake1 i (x:xs) | i > 0 = x : myTake1 (i-1) xs
                 | otherwise = []

-- | drop
--
-- >>> myDrop1 0 [] == drop 0 []
-- True
--
-- >>> myDrop1 10 [] == drop 10 []
-- True
--
-- >>> myDrop1 2 [1, 2, 3, 4] == drop 2 [1, 2, 3, 4]
-- True
--
-- >>> myDrop1 (-1) [] == drop (-1) []
-- True
--
-- >>> myDrop1 5 [1, 2] == drop 5 [1, 2]
-- True
myDrop1 :: Int -> [a] -> [a]
myDrop1 _ [] = []
myDrop1 i a@(_:xs) | i > 0 = myDrop1 (i-1) xs
                   | otherwise = a

myMinMax1 :: (Num a, Ord a) => (a -> a -> a) -> [a] -> a
myMinMax1 _ [] = errorEmptyList
myMinMax1 _ [x] = x
myMinMax1 f [x0, x1] = f x0 x1
myMinMax1 f (x0:x1:xs) = myMinMax1 f (f x0 x1:xs)

myMinMax2 :: (Num a, Ord a) => (a -> a -> a) -> [a] -> a
myMinMax2 _ [] = errorEmptyList
myMinMax2 _ [x] = x
myMinMax2 f (x:xs) = f x (myMinMax2 f xs)

-- | minimum
--
-- >>> minimum []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myMinimum1 []
-- *** Exception: empty list
-- ...
--
-- >>> myMinimum1 [1] == minimum [1]
-- True
--
-- >>> myMinimum1 [3, 1, 5] == minimum [3, 1, 5]
-- True
--
-- >>> myMinimum1 [8, 4, 2, 1, 5, 6] == minimum [8, 4, 2, 1, 5, 6]
-- True
--
-- >>> myMinimum1 [1, 5, 6] == minimum [1, 5, 6]
-- True
--
-- >>> myMinimum1 [5, 6, 1] == minimum [5, 6, 1]
-- True
--
-- >>> myMinimum2 []
-- *** Exception: empty list
-- ...
--
-- >>> myMinimum2 [1] == minimum [1]
-- True
--
-- >>> myMinimum2 [3, 1, 5] == minimum [3, 1, 5]
-- True
--
-- >>> myMinimum2 [8, 4, 2, 1, 5, 6] == minimum [8, 4, 2, 1, 5, 6]
-- True
--
-- >>> myMinimum2 [1, 5, 6] == minimum [1, 5, 6]
-- True
--
-- >>> myMinimum2 [5, 6, 1] == minimum [5, 6, 1]
-- True
myMinimum1 :: (Num a, Ord a) => [a] -> a
myMinimum1 = myMinMax1 myMin1
myMinimum2 :: (Num a, Ord a) => [a] -> a
myMinimum2 = myMinMax2 myMin1

-- | Maximum
--
-- >>> maximum []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myMaximum1 []
-- *** Exception: empty list
-- ...
--
-- >>> myMaximum1 [1] == maximum [1]
-- True
--
-- >>> myMaximum1 [3, 1, 5] == maximum [3, 1, 5]
-- True
--
-- >>> myMaximum1 [8, 4, 2, 1, 5, 6] == maximum [8, 4, 2, 1, 5, 6]
-- True
--
-- >>> myMaximum1 [1, 5, 6] == maximum [1, 5, 6]
-- True
--
-- >>> myMaximum1 [5, 6, 1] == maximum [5, 6, 1]
-- True
--
-- >>> myMaximum2 []
-- *** Exception: empty list
-- ...
--
-- >>> myMaximum2 [1] == maximum [1]
-- True
--
-- >>> myMaximum2 [3, 1, 5] == maximum [3, 1, 5]
-- True
--
-- >>> myMaximum2 [8, 4, 2, 1, 5, 6] == maximum [8, 4, 2, 1, 5, 6]
-- True
--
-- >>> myMaximum2 [1, 5, 6] == maximum [1, 5, 6]
-- True
--
-- >>> myMaximum2 [5, 6, 1] == maximum [5, 6, 1]
-- True
myMaximum1 :: (Num a, Ord a) => [a] -> a
myMaximum1 = myMinMax1 myMax1
myMaximum2 :: (Num a, Ord a) => [a] -> a
myMaximum2 = myMinMax2 myMax1

-- | sum
--
-- >>> mySum1 [] == sum []
-- True
--
-- >>> mySum1 [1] == sum [1]
-- True
--
-- >>> mySum1 [-3, 1, 7] == sum [-3, 1, 7]
-- True
--
-- >>> mySum2 [] == sum []
-- True
--
-- >>> mySum2 [1] == sum [1]
-- True
--
-- >>> mySum2 [-3, 1, 7] == sum [-3, 1, 7]
-- True
mySum1 :: (Num a) => [a] -> a
mySum1 [] = 0
mySum1 (x:xs) = x + mySum1 xs
{-# ANN mySum1 "HLint: ignore Use foldr" #-}
mySum2 :: (Num a) => [a] -> a
mySum2 = myFoldl1 (+) 0

-- | product
--
-- >>> myProduct1 [] == product []
-- True
--
-- >>> myProduct1 [1] == product [1]
-- True
--
-- >>> myProduct1 [-3, 1, 7] == product [-3, 1, 7]
-- True
--
-- >>> myProduct2 [] == product []
-- True
--
-- >>> myProduct2 [1] == product [1]
-- True
--
-- >>> myProduct2 [-3, 1, 7] == product [-3, 1, 7]
-- True
myProduct1 :: (Num a) => [a] -> a
myProduct1 [] = 1
myProduct1 (x:xs) = x * myProduct1 xs
{-# ANN myProduct1 "HLint: ignore Use foldr" #-}
myProduct2 :: (Num a) => [a] -> a
myProduct2 = myFoldl1 (*) 1

-- | elem
--
-- >>> myElem1 1 [] == elem 1 []
-- True
--
-- >>> myElem1 1 [2, 3, 1] == elem 1 [2, 3, 1]
-- True
--
-- >>> myElem1 1 [3] == elem 1 [3]
-- True
--
-- >>> myElem1 3 [1..] == elem 3 [1..]
-- True
--
myElem1 :: Eq a => a -> [a] -> Bool
myElem1 _ [] = False
myElem1 a (x:xs) | a == x = True
                 | otherwise = myElem1 a xs

myNotElem1 :: Eq a => a -> [a] -> Bool
myNotElem1 a = not . myElem1 a

-- | cycle
--
-- >>> cycle []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myCycle1 []
-- *** Exception: empty list
-- ...
--
-- >>> myTake1 10 (myCycle1 [1, 2, 3]) == take 10 (cycle [1, 2, 3])
-- True
--
--
-- >>> myTake1 10 (myCycle1 [1]) == take 10 (cycle [1])
-- True
myCycle1 :: [a] -> [a]
myCycle1 [] = errorEmptyList
myCycle1 a = a ++ myCycle1 a

-- | repeat
--
-- >>> myTake1 10 (myRepeat1 1) == take 10 (repeat 1)
-- True
--
-- >>> myTake1 10 (myRepeat1 []) == take 10 (repeat [])
-- True
myRepeat1 :: a -> [a]
myRepeat1 a = a:myRepeat1 a

-- | replicate
--
-- >>> myReplicate1 (-1) [] == replicate (-1) []
-- True
--
-- >>> myReplicate1 0 [] == replicate 0 []
-- True
--
-- >>> myReplicate1 3 'c' == replicate 3 'c'
-- True
--
-- >>> myReplicate1 3 [] == replicate 3 []
-- True
--
-- >>> myReplicate1 3 [1,2] == replicate 3 [1,2]
-- True
--
-- >>> myReplicate2 (-1) [] == replicate (-1) []
-- True
--
-- >>> myReplicate2 0 [] == replicate 0 []
-- True
--
-- >>> myReplicate2 3 'c' == replicate 3 'c'
-- True
--
-- >>> myReplicate2 3 [] == replicate 3 []
-- True
--
-- >>> myReplicate2 3 [1,2] == replicate 3 [1,2]
-- True
myReplicate1 :: Int -> a -> [a]
myReplicate1 i x | i > 0 = x:myReplicate1 (i-1) x
                 | otherwise = []
myReplicate2 :: Int -> a -> [a]
myReplicate2 i x = myTake1 i (myRepeat1 x)

-- | fst
--
-- >>> myFst1 (1, 2) == fst (1, 2)
-- True
myFst1 :: (a, b) -> a
myFst1 (a, _) = a

-- | snd
--
-- >>> mySnd1 (1, 2) == snd (1, 2)
-- True
mySnd1 :: (a, b) -> b
mySnd1 (_, b) = b

-- | zip
--
-- >>> myZip1 [] [] == zip [] []
-- True
--
-- >>> myZip1 [1, 2, 3] [4, 5, 6] == zip [1, 2, 3] [4, 5, 6]
-- True
--
-- >>> myZip1 [1, 2] [4, 5, 6] == zip [1, 2] [4, 5, 6]
-- True
--
-- >>> myZip1 [1, 2, 3] [4, 5] == zip [1, 2, 3] [4, 5]
-- True
--
-- >>> myTake1 3 (myZip1 [1, 2, 3] [4, 5]) == take 3 (zip [1, 2, 3] [4, 5])
-- True
myZip1 :: [a] -> [b] -> [(a, b)]
myZip1 _ [] = []
myZip1 [] _ = []
myZip1 (x:xs) (y:ys) = (x, y):myZip1 xs ys

-- | quick sort
--
-- >>> import Data.List (sort)
-- >>> myQSort1 [] == sort []
-- True
--
-- >>> myQSort1 [1] == sort [1]
-- True
--
-- >>> myQSort1 [1,2,3] == sort [1,2,3]
-- True
--
-- >>> myQSort1 [3,2,1] == sort [3,2,1]
-- True
--
-- >>> myQSort1 [3,5,7,9,0,4,2] == sort [3,5,7,9,0,4,2]
-- True
--
-- >>> myQSort2 [] == sort []
-- True
--
-- >>> myQSort2 [1] == sort [1]
-- True
--
-- >>> myQSort2 [1,2,3] == sort [1,2,3]
-- True
--
-- >>> myQSort2 [3,2,1] == sort [3,2,1]
-- True
--
-- >>> myQSort2 [3,5,7,9,0,4,2] == sort [3,5,7,9,0,4,2]
-- True
myQSort1 :: (Ord a) => [a] -> [a]
myQSort1 [] = []
myQSort1 (x:xs) = myQSort1 smallerOnes ++ [x] ++ myQSort1 largerOnes
  where smallerOnes = [i | i <- xs, i <  x]
        largerOnes  = [i | i <- xs, i >= x]
myQSort2 :: (Ord a) => [a] -> [a]
myQSort2 [] = []
myQSort2 (x:xs) = myQSort2 (myFilter1 (< x) xs) ++ [x] ++ myQSort2 (myFilter1 (>= x) xs)

-- | zipWith
--
-- >>> myZipWith1 (+) [1, 2, 3] [4, 5] == zipWith (+) [1, 2, 3] [4, 5]
-- True
myZipWith1 :: (a -> b -> c) -> [a] -> [b] -> [c]
myZipWith1 _ [] _ = []
myZipWith1 _ _ [] = []
myZipWith1 f (x:xs) (y:ys) = f x y:myZipWith1 f xs ys

-- | flip
--
-- >>> myFlip1 (-) 3 5 == flip (-) 3 5
-- True
--
-- >>> myFlip1 (:) "hello" 'c' == flip (:) "hello" 'c'
-- True
myFlip1 :: (b -> a -> c) -> a -> b -> c
myFlip1 f x y = f y x

-- | map
--
-- >>> myMap1 (1 +) [1,2,3] == map (1 +) [1,2,3]
-- True
--
-- >>> myMap1 id [] == map id []
-- True
--
-- >>> myMap2 (1 +) [1,2,3] == map (1 +) [1,2,3]
-- True
--
-- >>> myMap2 id [] == map id []
-- True
myMap1 :: (a -> b) -> [a] -> [b]
myMap1 _ [] = []
myMap1 f (x:xs) = f x:myMap1 f xs
myMap2 :: (a -> b) -> [a] -> [b]
myMap2 f = myFoldr1 (\x acc -> f x:acc) []

-- | filter
--
-- >>> myFilter1 (\x -> x > 0) [3, -1, 0, 7, -3] == filter (> 0) [3, -1, 0, 7, -3]
-- True
--
-- >>> myFilter1 (\_ -> True) [] == filter (\_ -> True) []
-- True
myFilter1 :: (a -> Bool) -> [a] -> [a]
myFilter1 _ [] = []
myFilter1 f (x:xs) | f x = x:myFilter1 f xs
                   | otherwise = myFilter1 f xs

-- | foldl
--
-- >>> myFoldl1 (+) 0 [1, 2, 3] == foldl (+) 0 [1,2,3]
-- True
--
-- >>> myFoldl1 (+) 0 [] == foldl (+) 0 []
-- True
myFoldl1 :: (a -> b -> a) -> a -> [b] -> a
myFoldl1 _ a [] = a
myFoldl1 f a (x:xs) = let newA = f a x
                      in  myFoldl1 f newA xs

-- | foldl1
--
-- >>> myFoldl11 (+) [1] == foldl1 (+) [1]
-- True
--
-- >>> myFoldl11 (+) [1, 2, 3] == foldl1 (+) [1,2,3]
-- True
--
-- >>> foldl1 (+) []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myFoldl11 (+) []
-- *** Exception: empty list
-- ...
myFoldl11 :: (a -> a -> a) -> [a] -> a
myFoldl11 _ [] = errorEmptyList
myFoldl11 f (x:xs) = myFoldl1 f x xs

-- | foldr
--
-- >>> myFoldr1 subtract 0 [1, 2, 3] == foldr subtract 0 [1,2,3]
-- True
--
-- >>> myFoldr1 subtract 0 [] == foldr subtract 0 []
-- True
myFoldr1 :: (b -> a -> a) -> a -> [b] -> a
myFoldr1 _ a [] = a
myFoldr1 f a (x:xs) = f x (myFoldr1 f a xs)

-- | foldr1
--
-- >>> myFoldr11 (+) [1] == foldr1 (+) [1]
-- True
--
-- >>> myFoldr11 (+) [1, 2, 3] == foldr1 (+) [1,2,3]
-- True
--
-- >>> foldr1 (+) []
-- *** Exception: ...: empty list
-- ...
--
-- >>> myFoldr11 (+) []
-- *** Exception: empty list
-- ...
myFoldr11 :: (a -> a -> a) -> [a] -> a
myFoldr11 _ [] = errorEmptyList
myFoldr11 _ [x] = x
myFoldr11 f (x:xs) = f x (myFoldr11 f xs)

-- | scanl
--
-- >>> myScanl1 (+) 0 [3,5,2,1] == scanl (+) 0 [3,5,2,1]
-- True
--
-- >>> myScanl1 (+) 2 [] == scanl (+) 2 []
-- True
myScanl1 :: (a -> b -> a) -> a -> [b] -> [a]
myScanl1 _ a [] = [a]
myScanl1 f a (x:xs) = a:myScanl1 f (f a x) xs

-- | scanl1
--
-- >>> myScanl11 (+) [3,5,2,1] == scanl1 (+) [3,5,2,1]
-- True
--
-- >>> myScanl11 (+) [] == scanl1 (+) []
-- True
myScanl11 :: (a -> a -> a) -> [a] -> [a]
myScanl11 _ [] = []
myScanl11 f (x:xs) = myScanl1 f x xs

-- | scanr
--
-- >>> myScanr1 subtract 10 [3,5,2] == scanr subtract 10 [3,5,2]
-- True
--
-- >>> myScanr1 subtract 0 [3,5,2,1] == scanr subtract 0 [3,5,2,1]
-- True
--
-- >>> myScanr1 subtract 2 [] == scanr subtract 2 []
-- True
myScanr1 :: (b -> a -> a) -> a -> [b] -> [a]
myScanr1 _ a [] = [a]
myScanr1 f a (x:xs) = let n = myScanr1 f a xs
                      in  f x (head n):n

-- | scanr1
--
-- >>> myScanr11 subtract [3,5,2,1] == scanr1 subtract [3,5,2,1]
-- True
--
-- >>> myScanr11 subtract [] == scanr1 subtract []
-- True
myScanr11 :: (a -> a -> a) -> [a] -> [a]
myScanr11 _ [] = []
myScanr11 _ [x0] = [x0]
myScanr11 f (x:xs) = let n = myScanr11 f xs
                     in f x (head n):n

