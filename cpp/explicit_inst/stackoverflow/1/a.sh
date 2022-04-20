# W : weak symbol

for i in *.o; do
	echo $i
	nm -CS $i | grep $1
done
echo a.out
nm -CS a.out |grep $1
