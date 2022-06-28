import re 
from collections import Counter

def words(text):
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('dictionary.txt').read()))
SIGNS = ['.', ',', '-', '?']

#probability of word
def P(word, N=sum(WORDS.values())):
    return WORDS[word] / N 

#the subset of word that appear in the dictionary of WORDS
def known(word):
    return set(w for w in word if w in WORDS)

#first edit 
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)

#second edit
def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

#generate possible spelling corrections for word 
def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

#most proable spelling correction for word 
def correction(word):
    if (len(word) == 0):
        return word 
    
    last_char = word[len(word) - 1]
    if word.istitle() or last_char in SIGNS:
        corrected_word = word
        if word.istitle() and last_char not in SIGNS:
            word = word.lower()
            corrected_word = max(candidates(word), key=P)
            return corrected_word.title()
        elif word.istitle() and last_char in SIGNS:
            word = word[0:len(word) - 1]
            word = word.lower()
            corrected_word = corrected_word.title()
            corrected_word = corrected_word + last_char 
            return corrected_word
        elif not word.istitle() and  last_char in SIGNS:
            word = word[0:len(word) - 1]  
            corrected_word = max(candidates(word), key=P)
            corrected_word = corrected_word + last_char 
            return corrected_word
    else:
        word = word.lower()
        return max(candidates(word), key=P)
