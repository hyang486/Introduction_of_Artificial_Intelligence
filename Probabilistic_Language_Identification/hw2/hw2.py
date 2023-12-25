import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment

    X = dict()
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        upperAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        eachLineinFIle = f.read()
        
        for word in upperAlphabet:
            X[word] = 0
        
        for wordInFile in eachLineinFIle:
            if wordInFile.islower():
                wordInFile = wordInFile.upper()
            
            if wordInFile in X:
                X[wordInFile] += 1

    return X

print("Q1")
dictFromFile = shred("letter.txt")
for i in dictFromFile:
    print(i , dictFromFile[i])
print("\n")

print("Q2")
frequency = dictFromFile['A']
english = frequency * math.log(get_parameter_vectors()[0][0])
espanol = frequency * math.log(get_parameter_vectors()[1][0])
print("%.4f" %english)
print("%.4f" %espanol)
print("\n")
 
print("Q3")
fenglish = 0.0
fespanol = 0.0
startchar = 'A'
for i in range(26):
    frequency2 = dictFromFile[startchar]
    fenglish += frequency2 * math.log(get_parameter_vectors()[0][i])
    fespanol += frequency2 * math.log(get_parameter_vectors()[1][i])
    startchar = chr(ord(startchar) + 1)
    
    
fEnglish = math.log(0.6) + fenglish
fEspanol = math.log(0.4) + fespanol

print("%.4f" %fEnglish)
print("%.4f" %fEspanol)
print("\n")  

print("Q4")
result = 0.0
if fEspanol - fEnglish >= 100: 
    result = 0
elif fEspanol - fEnglish <= -100: 
    result = 1
else:
    result = 1/(1 + math.exp(fEspanol - fEnglish))

print("%.4f" %result)
 

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

