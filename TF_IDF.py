from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from scipy import spatial as spatial

# this part defines path of data files
corpusPath = 'Dataset\Corpus2.txt'
queryPath = 'Dataset\Query.txt'
judgmentPath = 'Dataset\Judgement.txt'
resultPath = 'Dataset\Result.txt'
retrievalPath = 'Dataset\Retrieval2.txt'


# this func is used to parse the corpus file, it returns array of DID, Corpus, distinct words in all documents, and
# count of words in each document

def parseText(path):
    file = open(path, encoding='utf-8')
    lineNum = 0
    list = []
    doc_info = []
    begin = 0
    document = ""
    corpus = ""
    doc_length = []
    for line in file:
        lineNum += 1
        match = line.find('.DID')
        if match == 0:
            str = line.replace('.DID', '')
            str = str.replace('\t', '')
            str = str.replace('\n', '')
            doc_info.append(str.lower())
        if begin == 1:
            has_match = line.find('.DID')
            if has_match > -1:
                begin = 0
                list.append(document)
                corpus += document
                doc_length.append(len(nltk.word_tokenize(document)))
                document = ""
            else:
                document += line
        elif begin == 0:
            has_match = line.find('.Cat')
            if has_match > -1:
                begin = 1
    list.append(document)
    corpus += document
    nltk_tokens = nltk.word_tokenize(corpus)
    distinct = np.array(np.unique(nltk_tokens))
    doc_length.append(len(nltk.word_tokenize(document)))
    return list, doc_info, distinct, doc_length


# this function is used to extract each tag. given the name of tag it returns all the data within it
def extractTag(path, tag):
    begin_str = '<' + tag + '>'
    end_str = '</' + tag + '>'
    token = ""
    file = open(path, encoding='utf-8')
    begin = 0
    values = []
    for line in file:
        match = line.find(begin_str)
        match2 = line.find(end_str)
        if match == 0:
            begin = 1
        elif begin == 1 and match2 == -1:
            token += line
        if match2 == 0:
            begin = 0
            token = token.replace("\n", "")
            token = token.replace(".", " ")
            token.replace("،", " ")
            values.append(token)
            token = ""
    return values


# this func combine arrays of str
def combineArrays(title, description, narrative):
    res = []
    tmp = len(title)
    for i in range(0, tmp):
        str = title[i] + " "
        str += description[i] + " "
        str += narrative[i] + " "
        res.append(str)
    return res


# this function is used to parse Query file
def parseQuery(queryPath):
    QID = extractTag(queryPath, 'QID')
    title = extractTag(queryPath, 'title')
    description = extractTag(queryPath, 'description')
    narrative = extractTag(queryPath, 'narrative')
    main_text_query = combineArrays(title, description, narrative)
    return QID, title, main_text_query


# this func reads judgments from given file
def parseJudgment(path):
    judges = []
    file = open(path, encoding='utf-8')
    for line in file:
        words = nltk.word_tokenize(line)
        judges.append(words)
    return judges


# this function calculates IDF for each word in corpus and stores it in array called IDF
def calculate_IDF(distinct, main_text):
    IDF1 = np.zeros([len(distinct)])
    for i in range(len(distinct)):
        temp = 0
        for j in range(len(main_text)):
            temp += main_text[j].count(distinct[i])
        if temp != 0: IDF1[i] = np.math.log(float(len(main_text)) / temp)
    return IDF1


# this function calculates term frequency and stores it array called TF_IDF_array
# type true count - type false binary
def calculate_TF_IDF(text):
    TF_IDF_array = np.zeros([len(distinct), len(text)])
    docID = 0
    for doc in text:
        words = (word_tokenize(doc))
        l = len(words)
        for word in words:
            i, = np.where(distinct == word)
            TF_IDF_array[i, docID] += IDF[i]
        docID += 1
    return TF_IDF_array


# this function changes a numerical matrix to a binary ones.
def change_to_binary(array):
    result = np.zeros([len(array), len(array[0])])
    counter1 = 0
    for i in array:
        counter2 = 0
        for j in i:
            if j > 0: result[counter1, counter2] = IDF[counter2]
            counter2 += 1
        counter1 += 1
    return result


# this func returns norm of an array
def norm(a):
    return np.math.sqrt(np.dot(a, a))


# this func calculates cosine distance of two arrays
def cosine_distance(a, b):
    if norm(a) > 0 and norm(b) > 0:
        return np.dot(a, b) / (norm(a) * norm(b))
    else:
        return 0


# this func calculates jaccard distance of two arrays
def jaccard_distance(a, b):
    intersection = 0.0
    union = 0.0
    for i in range(len(a)):
        if a[i] > 0 and a[i] == b[i]: intersection += 1
        if a[i] > 0 or b[i] > 0: union += 1
    return intersection / union


# this function returns the related documents to given query terms
def Query(type_distance, q_vec, array_doc):
    # q_vec = array_query[:, counter]
    results = []
    index = 0
    for docID in DID:
        doc_vec = array_doc[:, index]
        if type_distance == "cosine":
            results.append((docID, cosine_distance(q_vec, doc_vec)))
        elif type_distance == "jaccard":
            results.append((docID, jaccard_distance(q_vec, doc_vec)))
        index += 1
    return sorted(results, key=lambda t: t[1])


# print 15 similar doc
def part_a(array_query, array_doc):
    distance = ["cosine", "jaccard"]
    for d in distance:
        index = 0
        for q_vec in array_query:
            print("15 doc similar to " + str(QID[index]) + "with " + str(d) + " distance is :")
            print(Query(d, q_vec, array_doc)[:15])
            index += 1


# # this func evaluates given results based on gold data with precision@k measure
# def evaluation(result, gold, k):
#     tmp = 0
#     if k > len(result): k = len(result)
#     for i in range(0, k):
#         if result[i][0] in gold:
#             tmp += 1
#         print(str(result[i][0]))
#     return (tmp / k)
#
#
# # this func returns the gold data related to given QID
# def getGold(QID):
#     row = np.shape(judge)[0]
#     goldData = []
#     for i in range(0, row):
#         if QID == judge[i][0]:
#             goldData.append(judge[i][1])
#     return goldData
#
#
# # this func writes results to given file
# def writeToFile(f, res):
#     f.write("[")
#     for ID, precision in res:
#         f.write("\n".join(["(%s , %s)" % (ID, precision)]))
#     f.write("]\n")
#     return
#
#
# # this function evaluates set of given queries with given parameters which are used as k in precision@k measure
# # and writes results to 'pathToWriteResults' file
# def evalquery(type_distance, precisions):
#     query_ID = 0
#     f = open(resultPath, 'a')
#     counter_query = 0
#     for query in main_text_query:
#         res = Query(type_distance, query, TF_IDF_array)
#         counter_query += 1
#         for precision in precisions:
#             precisionAtK = evaluation(res, getGold(QID[query_ID]), precision)
#             template = "precision @ %s  for QID %s  is: %s \n"
#             f.write("\n")
#             print(template % (precision, QID[query_ID], precisionAtK))
#             f.write(template % (precision, QID[query_ID], precisionAtK))
#             writeToFile(f, res[:precision])
#         query_ID += 1
#     return


# read files and parse them
main_text, DID, distinct, doc_length = parseText(retrievalPath)
main_text_corpus, DID_corpus, distinct_corpus, doc_length_corpus = parseText(corpusPath)
QID, title_query, main_text_query = parseQuery(queryPath)
judge = parseJudgment(judgmentPath)
print("parsing file finished")

# report size of file
n_docs = len(main_text)
distinct = np.transpose(list(distinct))
print("number of docs: " + str(n_docs))
print("number of distinct words: " + str(len(distinct)))
print("avg length of docs: " + str(sum(doc_length) / len(doc_length)))
print("doc with max length: " + str(DID[doc_length.index(max(doc_length))]))
print("doc with min length: " + str(DID[doc_length.index(min(doc_length))]))

# # calculate TF_IDF array for doc and query file
# IDF = calculate_IDF(distinct, main_text_corpus)
# print("calculating IDF finished")
# TF_IDF_array = calculate_TF_IDF( main_text)
# TF_IDF_array_query = calculate_TF_IDF( title_query)
# print("calculating TF-IDF finished")

# # evaluate queries using p@5 p@10 p@20
# print("start evaluation")
# precision = [5, 10]
# evalquery(precision)

# part a
print("compute 15 similar doc with cosine distance")
IDF = calculate_IDF(distinct, main_text_corpus)
print("calculating IDF finished")
TF_IDF_array = calculate_TF_IDF(main_text)
TF_IDF_array_query = calculate_TF_IDF(title_query)
TF_IDF_array_binary = change_to_binary(TF_IDF_array)
TF_IDF_array_query_binary = change_to_binary(TF_IDF_array_query)
print("calculating TF-IDF finished")
print("********numeric part********")
part_a(TF_IDF_array_query, TF_IDF_array)
print("********binary part********")
part_a(TF_IDF_array_query_binary, TF_IDF_array_binary)
