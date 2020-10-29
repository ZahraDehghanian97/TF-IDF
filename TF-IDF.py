import pandas as pd
import re


def find_between(s, start, end):
    print(s)
    print(s.split(start)[1])
    print((s.split(start))[1].split(end))
    print((s.split(start))[1].split(end)[0])
    return (s.split(start))[1].split(end)[0]


def importData(fileName):
    f = open(fileName, encoding='utf-8')
    result = re.search('<QID></QID>', f.read())
    print(result)
    # f = f.read().split("</narrative>")
    # QID = []
    # title = []
    # description = []
    # narrative = []
    # for item in f:
    #     QID.append(int(find_between(item, "<QID>", "</QID>")))
    # print(QID)


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


importData("Dataset\Query.txt")
# bowA = docA.split(" ")
# bowB = docB.split(" ")
# # make set of unique word of each files
# wordSet = set(bowA).union(set(bowB))
# wordDictA = dict.fromkeys(wordSet, 0)
# wordDictB = dict.fromkeys(wordSet, 0)
# # count the number of occurrences of each word
# for word in bowA:
#     wordDictA[word] += 1
#
# for word in bowB:
#     wordDictB[word] += 1
# # print the count of each word in each file
# print(pd.DataFrame([wordDictA, wordDictB]))
# tfBowA = computeTF(wordDictA, bowA)
# tfBowB = computeTF(wordDictB, bowB)
# idfs = computeIDF([wordDictA, wordDictB])
# tfidfBowA = computeTFIDF(tfBowA, idfs)
# tfidfBowB = computeTFIDF(tfBowB, idfs)
# print(pd.DataFrame([tfidfBowA, tfidfBowB]))
