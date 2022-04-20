foldl' :: (a -> a -> a) -> a -> [a] -> a
foldl' _ i [] = i
foldl' f i (x:xs) = foldl' f a xs where a = f i x

foldl1' :: (a -> a -> a) -> [a] -> a
foldl1' _ [] = error "empty list"
foldl1' f (x:xs) = foldl' f x xs

foldr' :: (a -> a -> a) -> a -> [a] -> a
foldr' _ i [] = i
foldr' f i (x:xs) = f (foldr' f i xs) x

foldr1' :: (a -> a -> a) -> [a] -> a
foldr1' _ [] = error "empty list"
foldr1' f (x:xs) = f x (foldr1' f xs)

scanl' :: (a->a->a) -> a -> [a] -> [a]
scanl' _ i [] = [i]
scanl' f i (x:xs) = i:scanl' f (f i x) xs

scanl1' :: (a->a->a) -> [a] -> [a]
scanl1' _ [] = error "empty list"
scanl1' f (x:xs) = scanl' f x xs

scanr' :: (a->a->a) -> a -> [a] -> [a]
scanr' _ i [] = [i]
scanr' f i (x:xs) = f x q : qs where qs@(q:_) = scanr' f i xs

scanr1' :: (a->a->a) -> [a] -> [a]
scanr1' _ [] = error "empty list"
scanr1' _ [x] = [x]
scanr1' f (x:xs) = f x q : qs where qs@(q:_) = scanr1' f xs

conc' :: [a] -> [a] -> [a]
conc' []     ys = ys
conc' (x:xs) ys = x:conc' xs ys

otherwise' :: Bool
otherwise' = True

at' :: (Integral i) => [a] -> i -> a
at' []     _ = error "out of bound"
at' (x:xs) i | i == 0 = x
             | i > 0  = at' xs (i-1)
             | i < 0  = error "out of bound"

length' :: [a] -> Integer
length' []     = 0
length' (_:xs) = 1 + length' xs

null' :: [a] -> Bool
null' [] = True
null' _  = False

reverse' :: [a] -> [a]
reverse' [] = []
-- reverse' (x:xs) = (reverse' xs) ++ [x]
reverse' l  = rev l []
  where rev []     a = a
        rev (x:xs) a = rev xs (x:a)

head' :: [a] -> a
head' []     = error "empty list"
head' (x:_) = x

tail' :: [a] -> [a]
tail' []     = error "empty list"
tail' (_:xs) = xs

init' :: [a] -> [a]
init' []     = error "empty list"
init' [x]    = []
init' (x:xs) = x:init' xs

last' :: [a] -> a
last' []     = error "empty list"
last' [x]    = x
last' (_:xs) = last' xs

take' :: (Integral i) => i -> [a] -> [a]
take' i _      | i <= 0 = []
take' _ []     = []
take' i (x:xs) = x:take' (i-1) xs

drop' :: (Integral i) => i -> [a] -> [a]
drop' i a | i <= 0 = a
drop' _ [] = []
drop' i (_:xs) = drop' (i-1) xs

max' :: (Ord a) => a -> a -> a
max' x y = if x > y then x else y

min' :: (Ord a) => a -> a -> a
min' x y = if x < y then x else y

minimum' :: (Ord a) => [a] -> a
--minimum' [] = error "empty list"
{-
minimum' [x] = x
minimum' (x:xs) = f x xs
  where f m [x]    = min' m x
        f m (x:xs) = f (f m [x]) xs
-}
--minimum' (x:xs) = foldl' min' x xs
minimum' = foldr1' min'

maximum' :: (Ord a) => [a] -> a
--maximum' [] = error "empty list"
{-
maximum' [x] = x
maximum' (x:xs) = f x xs
  where f m [x]    = max' m x
        f m (x:xs) = f (f m [x]) xs
-}
--maximum' (x:xs) = foldl' max' x xs
maximum' = foldr1' max'

sum' :: (Num a) => [a] -> a
--sum' [] = 0
--sum' (x:xs) = x + sum' xs
--sum' (x:xs) = foldl' (+) x xs
sum' = foldr1' (+)

product' :: (Num a) => [a] -> a
--product' [] = 1
--product' (x:xs) = x * product' xs
--product' (x:xs) = foldl' (*) x xs
product' = foldr1' (*)

elem' :: (Eq a) => a -> [a] -> Bool
elem' _ [] = False
elem' a (x:xs) = a == x || elem' x xs

cycle' :: [a] -> [a]
cycle' [] = error "empty list"
-- cycle' a = a ++ (cycle' a)
cycle' a = a' where a' = a ++ a'

repeat' :: a -> [a]
repeat' a = cycle' [a]

replicate' :: (Ord i, Num i) => i -> a -> [a]
replicate' i _ | i <= 0 = []
replicate' i a = a:(replicate' (i-1) a)

fst' :: (a, b) -> a
fst' (a, _) = a

snd' :: (a, b) -> b
snd' (_, b) = b

zip' :: [a] -> [b] -> [(a, b)]
zip' _ [] = []
zip' [] _ = []
zip' (a:as) (b:bs) = (a, b):(zip' as bs)

zipWith' :: (a->b->c) -> [a] -> [b] -> [c]
zipWith' _ _ [] = []
zipWith' _ [] _ = []
zipWith' f (a:as) (b:bs) = (f a b) : zipWith' f as bs

flip' :: (a -> b -> c) -> (b -> a -> c)
flip' f a b = f b a

map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x:map' f xs

filter' :: (a -> Bool) -> [a] -> [a]
filter' _ [] = []
filter' p (x:xs) = if p x then x:filter' p xs else filter' p xs

intersperse' :: a -> [a] -> [a]
intersperse' _ []     = []
intersperse' _ [x]    = [x]
intersperse' c (x:xs) = x:c:intersperse' c xs

intercalate' :: [a] -> [[a]] -> [a]
intercalate' _ [] = []
intercalate' _ [[]] = []
intercalate' _ (x:[]) = x
intercalate' as (x:xs) = x ++ as ++ (intercalate' as xs)

concat' :: [[a]] -> [a]
concat' [] = []
concat' [[]] = []
concat' (x:xs) = x ++ (concat' xs)

concatMap' :: (a -> [b]) -> [a] -> [b]
concatMap' f = concat' . map' f

transpose' :: [[a]] -> [[a]]
transpose' [] = []
transpose' ([]:xss) = transpose' xss
transpose' ((x:xs):xss) = (x:[h | (h:_) <- xss]):transpose' (xs : [t | (_:t) <- xss])

and' :: [Bool] -> Bool
and' [] = True
and' a = foldr1' (&&) a

or' :: [Bool] -> Bool
or' [] = False
or' a = foldr1' (||) a

any' :: (a -> Bool) -> [a] -> Bool
any' f a = or' . map' f $ a

all' :: (a -> Bool) -> [a] -> Bool
all' _ [] = True
all' f a = and' . map' f $ a

iterate'' :: (a -> a) -> a -> [a]
iterate'' f a = a : iterate f (f a)

splitAt :: Int -> [a] -> ([a], [a])
splitAt n a = (take' n a, drop' n a)

takeWhile' :: (a -> Bool) -> [a] -> [a]
takeWhile' _ [] = []
takeWhile' f (x:xs) | f x = x:takeWhile' f xs
                    | otherwise = []

dropWhile' :: (a -> Bool) -> [a] -> [a]
dropWhile' _ [] = []
dropWhile' f (x:xs) | f x = dropWhile' f xs
                    | otherwise = xs

span' :: (a -> Bool) -> [a] -> ([a], [a])
span' f xs = (takeWhile' f xs, dropWhile' f xs)

break' :: (a -> Bool) -> [a] -> ([a], [a])
break' f xs = span' (not . f) xs

sort' :: (Ord a) => [a] -> [a]
sort' (x:xs) = left ++ [x] ++ right
  where left = sort' $ filter' (< x) xs
        right = sort' $ filter' (>= x) xs

group' :: (Eq a) => [a] -> [[a]]
group' [] = []
group' [x] = [[x]]
group' (x:xs) = takeWhile' (==x) xs:(group' $ dropWhile' (==x) xs)
