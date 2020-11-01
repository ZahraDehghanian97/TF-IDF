from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from scipy import spatial as spatial

# this part defines path of data files
corpusPath = 'Dataset\Corpus.txt'
queryPath = 'Dataset\Query.txt'
judgmentPath = 'Dataset\Judgement.txt'
resultPath = 'Dataset\Result.txt'
retrievalPath = 'Dataset\Retrieval.txt'

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
    return QID, main_text_query


# this func reads judgments from given file
def parseJudgment(path):
    judges = []
    file = open(path, encoding='utf-8')
    for line in file:
        words = nltk.word_tokenize(line)
        judges.append(words)
    return judges


# this function calculates term frequency and stores it array called TF_IDF_array
def calculate_TF(type, normalize, text, array):
    TF_IDF_array = np.array(array)
    docID = 0
    flag = True
    if type == 2 : flag = False
    for doc in text:
        words = np.array(word_tokenize(doc))
        distinctID = 0
        for word in distinct:
            i = np.where(words == word)
            if flag :
                TF_IDF_array[distinctID, docID] = len(i[0])
            else :
                if len(i[0])>0 : TF_IDF_array[distinctID, docID] = 1
            if normalize == 1:
                TF_IDF_array[distinctID, docID] /= len(words)
            distinctID += 1
        docID += 1
    return TF_IDF_array


# this function calculates IDF for each word in corpus and stores it in array called IDF
def calculate_IDF(TF_IDF_array, main_text):
    col = np.shape(TF_IDF_array)[0]
    for term in range(0, col):
        df = np.count_nonzero(TF_IDF_array[term, :])
        if df != 0:
            df = np.math.log(float(len(main_text)) / df)
        IDF[term, 0] = df
    return


# this function calculates TF-IDF score and writes it in TF_IDF_array
# multiplies IDF of each word with it's TF
def calculate_TF_IDF(TF_IDF_array):
    cols = np.shape(TF_IDF_array)[1]
    rows = np.shape(TF_IDF_array)[0]
    for col in range(0, cols):
        for row in range(0, rows):
            TF_IDF_array[row, col] = TF_IDF_array[row, col] * (IDF[row])
    return TF_IDF_array


# this func returns the corresponding IDF array of given query terms.
def query_vector(counter):
    query_vector = TF_IDF_array_query[:, counter]
    return query_vector


# this function returns the related documents to given query terms
def Query(counter):
    q_vec = query_vector(counter)
    n_terms, _ = TF_IDF_array.shape
    results = []
    index = 0
    for doc in DID:
        doc_vec = TF_IDF_array[:, index]
        results.append((doc, spatial.distance.cosine(q_vec, doc_vec)))
        index += 1
    return sorted(results, key=lambda t: t[1], reverse=True)


# this func evaluates given results based on gold data with precision@k measure
def evaluation(result, gold, k):

    tmp = 0
    for i in range(0, k):
        if result[i][0] in gold:
            tmp += 1
        print( str(result[i][0]))
    return (tmp / k)


# this func returns the gold data related to given QID
def getGold(QID):
    row = np.shape(judge)[0]
    goldData = []
    for i in range(0, row):
        if QID == judge[i][0]:
            goldData.append(judge[i][1])
    return goldData


# this func writes results to given file
def writeToFile(f, res):
    f.write("[")
    for ID, precision in res:
        f.write("\n".join(["(%s , %s)" % (ID, precision)]))
    f.write("]\n")
    return


# this function evaluates set of given queries with given parameters which are used as k in precision@k measure
# and writes results to 'pathToWriteResults' file
def evalquery(precisions):
    query_ID = 0
    f = open(resultPath, 'a')
    counter_query = 0
    for query in main_text_query:
        res = Query(counter_query)
        counter_query += 1
        for precision in precisions:
            precisionAtK = evaluation(res, getGold(QID[query_ID]), precision)
            template = "precision @ %s  for QID %s  is: %s \n"
            f.write("\n")
            print(template % (precision, QID[query_ID], precisionAtK))
            f.write(template % (precision, QID[query_ID], precisionAtK))
            writeToFile(f, res[:20])
        query_ID += 1
    return


# read files and parse them
main_text, DID, distinct, doc_length = parseText(retrievalPath)
main_text_corpus, DID_corpus, distinct_corpus, doc_length_corpus = parseText(corpusPath)
QID, main_text_query = parseQuery(queryPath)
judge = parseJudgment(judgmentPath)
print("parsing file finished")

# create a TF-IDF array for document and query
TF_IDF_array = np.zeros([len(distinct), np.shape(DID)[0]])
TF_array_corpus = np.zeros([len(distinct), np.shape(DID_corpus)[0]])
TF_IDF_array_query = np.zeros([len(distinct), np.shape(QID)[0]])
IDF = np.zeros([len(distinct), 1])
print("built array to save result")

# calculate TF_IDF array for doc and query file
type = 1  # 1 == count term frequency  2 == binary term frequency
normalize = 1  # 0 == not normalize tf   1== normalize tf
TF_IDF_array = calculate_TF(type, normalize, main_text, TF_IDF_array)
TF_IDF_array_query = calculate_TF(type, normalize, main_text_query, TF_IDF_array_query)
print("calculating TF finished")
TF_array_corpus = calculate_TF(2, 0, main_text_corpus, TF_array_corpus)
calculate_IDF(TF_array_corpus, main_text_corpus)
print("calculating IDF finished")
TF_IDF_array = calculate_TF_IDF(TF_IDF_array)
TF_IDF_array_query = calculate_TF_IDF(TF_IDF_array_query)
print("calculating TF_IDF finished")

# evaluate queries using p@5 p@10 p@20
print("start evaluation")
precision = [5, 10, 20]
evalquery(precision)
