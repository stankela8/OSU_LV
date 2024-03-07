fhand = open ('example.txt')
for line in fhand :
    line = line . rstrip ()
    print ( line )
    words = line . split ()
fhand . close ()