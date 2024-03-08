'''
print("Unesi broj radnih sati: ")
radniSati=float(input())
print("Unesi satnicu: ")
satnica=float(input())
ukupno=radniSati*satnica
print("Ukupno: ",ukupno," eura")

def total_euro(x,y):
    return x*y

print("ukupno preko funkcije: ",total_euro(radniSati,satnica)," eura")
'''

#Drugi
'''
try:
    ocjena=float(input())
    if 0.0 <= ocjena <=1.0 :
        if ocjena >= 0.9 :
            kategorija='A'
        elif ocjena>=0.8:
            kategorija='B'
        elif ocjena>=0.7:
            kategorija='C'
        elif ocjena>=0.6:
            kategorija='D'
        else:
            kategorija='F'
        print(kategorija)
    else:
        print("Broj nije u intervalu")
except:
    print("Niste unijeli broj")
'''
#treci
'''
brojevi=[]

while True:
    unos=input("Unesite broj ili 'done': ")

    if unos.lower()=='done':
        break

    try:
        broj=float(unos)
        brojevi.append(broj)

    except:
        print("neispravan unos")

if brojevi:
    print("unjeli ste ",len(brojevi)," brojeva.")
    print("Srednja vrijednost unesenih brojeva je: ",sum(brojevi)/len(brojevi))
    print("maksimalna vrijednost: ",max(brojevi))
    print("minimalna vrijednost: ",min(brojevi))
    brojevi.sort()
    print(brojevi)
else:
    print("prazna lista")
'''

#cetvrti
rjecnik={}
''''
fhand=open("C:/Users/student/Desktop/LV1 Uvod u programski jezik Python-20240308/song.txt")
for line in fhand :
    words = line . split ()
    for word in words:
        rijec=word.lower()
        if rijec in rjecnik:
            rjecnik[rijec]+=1
        else:
            rjecnik[rijec]=1
rijeciJednom=[word for word,count in rjecnik.items() if count==1]

print("Broj rjeci koje se pojavljuju jednom: ",len(rijeciJednom))
print(rijeciJednom)

fhand . close ()
'''

#peti
counterHam=0
counterSpam=0
zavrsavaUsklicnikom=0
duzinaHam=0
duzinaSpam=0
fhand=open("C:/Users/student/Desktop/LV1 Uvod u programski jezik Python-20240308/SMSSpamCollection.txt")
for line in fhand:
    line=line.rstrip()
    words=line.split()
    #A
    if line.startswith('ham'):
        counterHam+=1
    else:
        counterSpam+=1


    if line.startswith('ham'):
        duzinaHam+=len(words)
    else:
        duzinaSpam+=len(words)

    #B
    if line.endswith('!') and line.startswith('spam'):
        zavrsavaUsklicnikom+=1

print(zavrsavaUsklicnikom)
print("prosjecan broj u hamu: ",round(duzinaHam/counterHam))
print("prosjecan broj u spamu: ",round(duzinaSpam/counterSpam))