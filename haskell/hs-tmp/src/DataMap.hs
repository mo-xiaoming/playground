module DataMap(
myFindKey1,
myFindKey2,
) where

-- | findKey
--
-- >>> phoneBook = [("betty","555-2938") ,("bonnie","452-2928") ,("patsy","493-2928") ,("lucille","205-2928") ,("wendy","939-8282") ,("penny","853-2492")]
-- >>> myFindKey1 "bonnie" phoneBook
-- Just "452-2928"
-- >>> myFindKey1 "hello" phoneBook
-- Nothing
myFindKey1 :: (Eq k) => k -> [(k,v)] -> Maybe v
myFindKey1 _ [] = Nothing
myFindKey1 k ((k0,v0):xs) = if k == k0 then Just v0 else myFindKey1 k xs

-- | findKey
--
-- >>> phoneBook = [("betty","555-2938") ,("bonnie","452-2928") ,("patsy","493-2928") ,("lucille","205-2928") ,("wendy","939-8282") ,("penny","853-2492")]
-- >>> myFindKey2 "bonnie" phoneBook
-- Just "452-2928"
-- >>> myFindKey2 "hello" phoneBook
-- Nothing
myFindKey2 :: (Eq k) => k -> [(k,v)] -> Maybe v
myFindKey2 k = foldr (\(k0,v0) acc -> if k == k0 then Just v0 else acc) Nothing

