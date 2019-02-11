# This work is licensed under the Creative Commons Attribution-Noncommercial-Share Alike 3.0 United States License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/us/ or send a letter to
# Creative Commons, 171 Second Street, Suite 300, San Francisco, California, 94105, USA.

# written by Jeff Huang
# cite: Huang, J. and Efthimiadis, E. N. 2009. Analyzing and evaluating query reformulation strategies in web search logs. In Proceeding of CIKM '09, 77-86.
# http://jeffhuang.com/cikm09_final.pdf

# v1.1
# Changes:
# 1) url stripping does more top-level domains
# 2) additional abbreviations are also detected
# 3) word substitution matches using path similarity and updated wordnet db
# Requires: python 2.6 and latest libraries listed below


from __future__ import division
import sys
import os
import time
import random

# Third Party Libraries
import numpy # http://numpy.scipy.org/
from nltk.corpus import wordnet # http://www.nltk.org/download    Wordnet is a separate download from the same site

from .porter import PorterStemmer
# The first argument to this program is the input file


# create the abbrevation mapping from the file called 'abbrev'
abbrev = {}
# for line in open('abbrev'):
#     (abbreviation, expandeds) = line.split(' - ')
#     abbrev[abbreviation] = expandeds.split(', ')

# Reformulation: Spelling Correction
# Returns levenshtein edit distance
def spellingCorrection(a, b):
    table = numpy.zeros((len(a)+1, len(b)+1))
    for i in range(len(a)+1):
        table[i, 0] = i
    for j in range(len(b)+1):
        table[0, j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1
            table[i,j] = min(table[i-1,j]+1, table[i,j-1]+1, table[i-1,j-1]+cost)

    return table[len(a), len(b)]

pscache = synsetcache = cached_word_similarity = {}

def querySimilarity(a, b):
    global pscache
    global synsetcache
    global iterations
    global cached_word_similarity
    iterations = 0
    sim = querySimilarityHelper(a.split(), b.split()) / max(len(a.split()), len(b.split()))
    if random.random() > 0.999:
        del synsetcache
        del pscache
        del cached_word_similarity
        pscache = {}
        synsetcache = {}
        cached_word_similarity = {}
    return sim

def querySimilarityHelper(a, b):
    global iterations
    if iterations >= 10000: # so we don't go on forever
        return 0
    iterations += 1
    if not a or not b:
        return 0
    maxSim = 0
    for ai in range(len(a)):
        for bi in range(len(b)):
            sim = querySimilarityHelper(a[:ai] + a[ai+1:], b[:bi] + b[bi+1:]) + bestPathSimilarity(a[ai], b[bi])
            if sim > maxSim:
                maxSim = sim
    return maxSim

def bestPathSimilarity(c, d):
    global synsetcache
    global cached_word_similarity
    if c == d:
        return 1

    if (c, d) in cached_word_similarity:
        return cached_word_similarity[(c, d)]

    maxPS = 0
    if c in synsetcache:
        csynset = synsetcache[c]
    else:
        csynset = wordnet.synsets(c)
        synsetcache[c] = csynset
    if d in synsetcache:
        dsynset = synsetcache[d]
    else:
        dsynset = wordnet.synsets(d)
        synsetcache[d] = dsynset
    for synC in csynset:
        for synD in dsynset:
            ps = cached_path_similarity(synC, synD)
            if ps > maxPS:
                maxPS = ps
    cached_word_similarity[(c, d)] = maxPS
    return maxPS

def cached_path_similarity(synA, synB):
    global pscache
    if (synA, synB) in pscache:
        return pscache[(synA, synB)]
    if (synB, synA) in pscache:
        return pscache[(synB, synA)]
    sim = synA.path_similarity(synB)
    sim = 0 if sim is None else sim
    pscache[(synA, synB)] = sim
    return sim

def wordSubstitutionHelper(a, b):
    for synA in wordnet.synsets(a):
        for synB in wordnet.synsets(b):
            real_sim = synA.path_similarity(synB)
            if not real_sim:
                continue
            if real_sim >= 0.2:
                return True

    return False

# Reformulation: Word Substitution
def wordSubstitution(a, b):
    aWords = a.split()
    bWords = b.split()

    if wordSubstitutionHelper(a, b):
        return True

    if len(aWords) != len(bWords):
        return False

    for i in range(len(aWords)):
        if aWords[i] == bWords[i]:
            continue
        if wordSubstitutionHelper(aWords[i], bWords[i]):
            continue
        return False
    return True

# Reformulation: Acronym
def formAcronym(a, b):
    if len(a.split()) <= 1 and len(b.split()) <= 1:
        return False
    if b.replace(".", "").replace("-", "") == "".join(i[0] for i in a.split()):
        return True
    return False

# Reformulation: Word Reorder
def wordReorder(a, b):
    aWords = []
    bWords = []
    for split1 in a.split():
        for split2 in split1.split('-'):
            for split3 in split2.split(','):
                aWords.append(split3)
    for split1 in b.split():
        for split2 in split1.split('-'):
            for split3 in split2.split(','):
                bWords.append(split3)
    aWords.sort()
    bWords.sort()
    return aWords == bWords

# Reformulation: Add Words, Remove Words
def addWords(a, b):
    aWords = a.split()
    bWords = b.split()
    for aWord in aWords:
        if aWord in bWords:
            continue
        return False
    return True

# Reformulation: Abbreviation
def abbreviation(a, b):
    aWords = a.split()
    bWords = b.split()
    if len(aWords) != len(bWords):
        return False

    for i in range(len(aWords)):
        if not (aWords[i].startswith(bWords[i]) or bWords[i].startswith(aWords[i]) or (aWords[i] in abbrev and bWords[i] in abbrev[aWords[i]]) or (bWords[i] in abbrev and aWords[i] in abbrev[bWords[i]])):
            return False
    return True

# Reformulation: Stemming
def stemming(a, b):
    p = PorterStemmer()
    aWords = a.split()
    bWords = b.split()
    if len(aWords) != len(bWords):
        return False

    for i in range(len(aWords)):
        if p.stem(aWords[i], 0, len(aWords[i])-1) != p.stem(bWords[i], 0, len(bWords[i])-1):
            return False
    return True

# Reformulation: URL Stripping
def urlStrip(a, b):
    aWords = a.split()
    bWords = b.split()

    done = False
    while not done:
        try:
            aWords.remove('http')
        except ValueError:
            done = True

    done = False
    while not done:
        try:
            bWords.remove('http')
        except ValueError:
            done = True

    if len(aWords) != len(bWords):
        return False

    for i in range(len(aWords)):
        if aWords[i].startswith('www.'):
            aWords[i] = aWords[i][4:]
        if aWords[i].endswith('.com'):
            aWords[i] = aWords[i][:-4]
        if aWords[i].endswith('.net'):
            aWords[i] = aWords[i][:-4]
        if aWords[i].endswith('.org'):
            aWords[i] = aWords[i][:-4]
        if aWords[i].endswith('.info'):
            aWords[i] = aWords[i][:-4]
        if aWords[i].endswith('.biz'):
            aWords[i] = aWords[i][:-4]
        if aWords[i].endswith('.gov'):
            aWords[i] = aWords[i][:-4]
        if aWords[i].endswith('.mil'):
            aWords[i] = aWords[i][:-4]
        if aWords[i].endswith('.eu'):
            aWords[i] = aWords[i][:-3]
        if aWords[i].endswith('.cn'):
            aWords[i] = aWords[i][:-3]
        if aWords[i].endswith('.de'):
            aWords[i] = aWords[i][:-3]
        if aWords[i].endswith('.uk'):
            aWords[i] = aWords[i][:-3]
        if aWords[i].endswith('.nl'):
            aWords[i] = aWords[i][:-3]

        if bWords[i].startswith('www.'):
            bWords[i] = bWords[i][4:]
        if bWords[i].endswith('.com'):
            bWords[i] = bWords[i][:-4]
        if bWords[i].endswith('.net'):
            bWords[i] = bWords[i][:-4]
        if bWords[i].endswith('.org'):
            bWords[i] = bWords[i][:-4]
        if bWords[i].endswith('.info'):
            bWords[i] = bWords[i][:-4]
        if bWords[i].endswith('.biz'):
            bWords[i] = bWords[i][:-4]
        if bWords[i].endswith('.gov'):
            bWords[i] = bWords[i][:-4]
        if bWords[i].endswith('.mil'):
            bWords[i] = bWords[i][:-4]
        if bWords[i].endswith('.eu'):
            bWords[i] = bWords[i][:-3]
        if bWords[i].endswith('.cn'):
            bWords[i] = bWords[i][:-3]
        if bWords[i].endswith('.de'):
            bWords[i] = bWords[i][:-3]
        if bWords[i].endswith('.uk'):
            bWords[i] = bWords[i][:-3]
        if bWords[i].endswith('.nl'):
            bWords[i] = bWords[i][:-3]

        if aWords[i] != bWords[i]: # might want to do compounding here too
            return False

    return True

# Reformulation: Whitespace and Punctuation
def whitespacePunctuation(a, b):
    return a.replace(" ", "").replace(".", "").replace("-", "") == b.replace(" ", "").replace(".", "").replace("-", "")


def tag(prev_query, query):
    if prev_query == query:
        type = 'same'
    elif wordReorder(prev_query, query):
        type = 'wordReorder'
    elif whitespacePunctuation(prev_query, query):
        type = 'whitespacePunctuation'
    elif addWords(prev_query, query):
        type = 'addWords'
    elif addWords(query, prev_query): # this is removeWords
        type = 'removeWords'
    elif urlStrip(prev_query, query):
        type = 'urlStrip'
    elif stemming(prev_query, query):
        type = 'stemming'
    elif formAcronym(prev_query, query):
        type = 'formAcronym'
    elif formAcronym(query, prev_query): # this is expandAcronym
        type = 'expandAcronym'
    elif abbreviation(prev_query, query): # this is a single metric because you can abbreviate terms in either query at the same time
        type = 'abbreviation'
    # Reformulation: Substring
    elif prev_query.startswith(query) or prev_query.endswith(query):
        type = 'substring'
    # Reformulation: Superstring
    elif query.startswith(prev_query) or query.endswith(prev_query):
        type = 'superstring'
    elif wordSubstitution(prev_query, query):
        type = 'wordSubstitution'
    elif spellingCorrection(prev_query, query) <= 2:
        type = 'spellingCorrection'
    else:
        type = 'new'
    return type


# CODE STARTS RUNNING HERE

if __name__ == '__main__':
    prevUserId = None
    prev_query = None
    prevURL = None
    prevRank = None
    prevTimestamp = None
    data = {}
    data['same'] = data['wordReorder'] = data['whitespacePunctuation'] = data['addWords'] = data['removeWords'] = data['urlStrip'] = data['stemming'] = data['wordSubstitution'] = data['formAcronym'] = data['expandAcronym']= data['abbreviation'] = data['spellingCorrection'] = data['substring'] = data['superstring'] = data['new'] = 0
    for line in open(sys.argv[1]):
        #print "LINE:", line.strip()
        columns = line.split("\t")
        try:
            userId = int(columns[0])
        except ValueError:
            continue

        query = columns[1]
        timestamp = columns[2]
        if columns[3]:
            try: # there is one bad line :(
                rank = int(columns[3])
            except ValueError:
                continue
        else:
            rank = None
        url = columns[4].strip()

        if query == "-" or prev_query == "-" or userId != prevUserId:
            prevUserId = userId
            prev_query = query
            prevTimestamp = timestamp
            prevRank = rank
            prevUrl = url
            continue

        if url and prevUrl:
            urlSame = (url == prevUrl)
        else:
            urlSame = None
        if rank and prevRank:
            rankChange = prevRank - rank # rankChange is how much the second query click went UP
        else:
            rankChange = None

        if rank and prevRank:
            clickPattern = "ClickClick"
        elif rank and not prevRank:
            clickPattern = "SkipClick"
        elif not rank and prevRank:
            clickPattern = "ClickSkip"
        else:
            clickPattern = "SkipSkip"

        timeDiff = time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(prevTimestamp, "%Y-%m-%d %H:%M:%S"))

        type = tag(prev_query, query)

        prev_query = query
        prevTimestamp = timestamp
        prevRank = rank
        prevUrl = url

        print(type, ',', userId, ',', int(timeDiff), ',', urlSame, ',', rankChange, ',', clickPattern)