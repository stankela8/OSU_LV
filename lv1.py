#Prvi
'''
x=float(input())
y=float(input())
rez=x*y
print("Radni sati: ",x)
print("eura/h: ",y)
print("Ukupno:",rez)

def total_euro(x,y):
    return x*y

rez2=total_euro(x,y)
print("Ukupno funkcija: ",rez2)
'''
#Drugi
'''
try:
    score = float(input("Unesite ocjenu između 0.0 i 1.0: "))

    if 0.0 <= score <= 1.0:
        if score >= 0.9:
            grade = 'A'
        elif score >= 0.8:
            grade = 'B'
        elif score >= 0.7:
            grade = 'C'
        elif score >= 0.6:
            grade = 'D'
        else:
            grade = 'F'
        print("Ocjena je: ",grade)
    else:
        print("Broj mora biti u intervalu od 0.0 do 1.0.")

except ValueError:
    print("Greška: Unesite brojčanu vrijednost.")

'''
#Treci
'''
numbers = []

while True:
    entry = input("Unesite broj ili 'Done' za završetak: ")
    
    if entry.lower() == 'done':
        break
    
    try:
        number = float(entry)
        numbers.append(number)
    except ValueError:
        print("Neispravan unos, molimo unesite broj ili 'Done'.")

if numbers:
    print(f"Unijeli ste ukupno {len(numbers)} brojeva.")
    print(f"Srednja vrijednost: {sum(numbers) / len(numbers)}")
    print(f"Minimalna vrijednost: {min(numbers)}")
    print(f"Maksimalna vrijednost: {max(numbers)}")
    numbers.sort()
    print("Sortirana lista brojeva:", numbers)
else:
    print("Niste unijeli nijedan broj.")
'''

#Cetvrti
# Učitavanje datoteke i stvaranje rječnika
word_count = {}

try:
    with open('C:/Users/lukas/Desktop/OSU/song.txt', 'r') as file:
        for line in file:
            words = line.split()
            for word in words:
                cleaned_word = word.lower().strip(".,!?\"'")
                if cleaned_word in word_count:
                    word_count[cleaned_word] += 1
                else:
                    word_count[cleaned_word] = 1
except FileNotFoundError:
    print("Datoteka 'song.txt' nije pronađena.")

# Pronalaženje i ispisivanje riječi koje se pojavljuju samo jednom
single_occurrence_words = [word for word, count in word_count.items() if count == 1]

print(f"Broj riječi koje se pojavljuju samo jednom: {len(single_occurrence_words)}")
print("Riječi koje se pojavljuju samo jednom:", single_occurrence_words)


