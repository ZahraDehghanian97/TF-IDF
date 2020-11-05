from nltk.tokenize import word_tokenize
import nltk
import numpy as np

# this part defines path of data files
corpusPath = 'Dataset\Corpus.txt'
queryPath = 'Dataset\Query.txt'
judgmentPath = 'Dataset\Judgement.txt'
retrievalPath = 'Dataset\Retrieval.txt'
result_APath = 'Dataset\judge_a.txt'
result_BPath = 'Dataset\judge_b.txt'


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
            token = token.replace("\t", '')
            token = token.replace("\n", '')
            token = token.replace(".", '')
            token.replace("،", '')
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
    l = 0
    for j in range(len(main_text)): l += len(main_text[j])
    for i in range(len(distinct)):
        temp = 0
        for j in range(len(main_text)):
            temp += main_text[j].count(distinct[i])
        if temp != 0:
            IDF1[i] = np.math.log(float(l) / temp)
    return IDF1


# this function calculates term frequency and stores it array called TF_IDF_array
def calculate_TF_IDF(text):
    TF_IDF_array1 = np.zeros([len(text), len(distinct)])
    docID = 0
    for doc in text:
        words = (word_tokenize(doc))
        for word in words:
            i, = np.where(distinct == word)
            TF_IDF_array1[docID, i] += IDF[i]
        docID += 1
    return TF_IDF_array1


# this function changes a numerical matrix to a binary ones.
def change_to_binary(array):
    result = np.zeros([len(array), len(array[0])])
    counter1 = 0
    for i in array:
        counter2 = 0
        for j in i:
            if j > 0:
                result[counter1, counter2] = IDF[counter2]
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
        doc_vec = array_doc[index]
        if type_distance == "cosine":
            results.append((docID, cosine_distance(q_vec, doc_vec)))
        elif type_distance == "jaccard":
            results.append((docID, jaccard_distance(q_vec, doc_vec)))
        index += 1
    return sorted(results, key=lambda t: t[1], reverse=True)


# print 15 similar doc with TF-IDF
def part_a(array_query, array_doc):
    distance = ["cosine", "jaccard"]
    for d in distance:
        index = 0
        print(str(d))
        for q_vec in array_query:
            result =(Query(d, q_vec, array_doc)[:15])
            for r in result:
                print(str(QID[index]) + "\t" + str(r[0]))
            index += 1


# this function run BM25 algorithm and return sorted list of doc
def BM25(b, k, q_vec, array_doc):
    results = []
    index = 0
    l_avg = np.average(doc_length)
    kplus = k + 1
    k2 = k * (1 - b)
    index_nonzero_query = np.nonzero(q_vec)[0]
    for docID in DID:
        doc_vec = array_doc[index]
        temp = doc_length[index]
        lastk = k2 + b * (temp / l_avg)
        result = 0
        for j in index_nonzero_query:
            r = kplus * doc_vec[j]
            t = doc_vec[j] / IDF[j]
            lastk += t
            r /= lastk
            result += r
        results.append((docID, result))
        index += 1
    return sorted(results, key=lambda t: t[1], reverse=True)


# find 15 similar doc with BM25
def part_b(array_query, array_doc):
    b = [0.5, 0.75, 1]
    k = [1.2, 1.5, 2]
    for b1 in b:
        for k1 in k:
            index = 0
            print("b = " + str(b1) + " k = " + str(k1))
            for q_vec in array_query:
                result = (BM25(b1, k1, q_vec, array_doc)[:15])
                for r in result:
                    print(str(QID[index]) + "\t" + str(r[0]))
                index += 1


# this function read result of each part and put them in arrays
def parseResult(path):
    result = []
    file = open(path)
    index = 0
    name = []
    group = []
    for line in file:
        if index == 76:
            index = 0
            result.append(group)
        if index == 0:
            t = line.replace("\n",'')
            name.append(t)
            group = []
        else:
            group.append(nltk.word_tokenize(line))
        index += 1
    result.append(group)
    return name, result


# this func returns the gold data related to given QID
def getID(list, QID):
    Data = []
    for i in range(0,len(list)):
        if QID == int(list[i][0]):
            Data.append(list[i][1])
    return Data


# this function evaluates precision of list 
def precision(list, p):
    result = []
    for qid in QID:
        my_result = getID(list, qid)
        gold_result = getID(judge, qid)
        tmp = 0
        for i in range(0, p):
            if my_result[i] in gold_result:
                tmp += 1
        result.append(tmp / p)
    return result


# this function compute MRR of list (sigma (1/first place relevant doc appear))
def MRR(list):
    result = 0
    for qid in QID:
        my_result = getID(list, qid)
        gold_result = getID(judge, qid)
        for i in range(0, len(my_result)):
            if my_result[i] in gold_result:
                result += (1 / (i + 1))
                break
    return result / len(QID)


# this function return Mean Average Precision for list
def MAP(list):
    result = 0
    for qid in QID:
        temp_result = 0
        my_result = getID(list, qid)
        gold_result = getID(judge, qid)
        index = 1
        true_match = 1
        for i in range(0, len(my_result)):
            if my_result[i] in gold_result:
                temp_result += (true_match / index)
                true_match += 1
            index += 1
        result += (temp_result / true_match)
    return result / len(QID)


# this function compute p@5 p@10 MAP MRR
def part_c(name, input):
    p_5 = []
    p_10 = []
    map = []
    mrr = []
    for group in input:
        p_5.append(precision(group, 5))
        p_10.append(precision(group, 10))
        map.append(MAP(group))
        mrr.append(MRR(group))
    temp = [p_5, p_10, map, mrr]
    name_metrics = ["p@5","p@10","MAP","MRR"]
    counter =0
    for t in temp:
        print("<<------------ "+str(name_metrics[counter])+" ------------>>")
        for index in range(len(name)):
            print(str(name[index]) + " : " + str(t[index]))
        counter+=1

# read files and parse them
main_text, DID, distinct, doc_length = parseText(retrievalPath)
main_text_corpus, DID_corpus, distinct_corpus, doc_length_corpus = parseText(corpusPath)
QID, title_query, main_text_query = parseQuery(queryPath)
print("parsing file finished")
# report some info about file
n_docs = len(main_text)
distinct = np.transpose(list(distinct))
print("number of docs: " + str(n_docs))
print("number of distinct words: " + str(len(distinct)))
print("avg length of docs: " + str(sum(doc_length) / len(doc_length)))
print("doc with max length: " + str(DID[doc_length.index(max(doc_length))]))
print("doc with min length: " + str(DID[doc_length.index(min(doc_length))]))

# compute TF-IDF array
IDF = calculate_IDF(distinct, main_text_corpus)
print("calculating IDF finished")
TF_IDF_array = calculate_TF_IDF(main_text)
TF_IDF_array_query = calculate_TF_IDF(main_text_query)
TF_IDF_array_binary = change_to_binary(TF_IDF_array)
TF_IDF_array_query_binary = change_to_binary(TF_IDF_array_query)
print("calculating TF-IDF finished")

# part a
print("<<<<<<<--------------- part a ------------------->>>>>>>")
print("compute 15 similar with TF-IDF model")
print("********numeric part********")
part_a(TF_IDF_array_query, TF_IDF_array)
print("********binary part********")
part_a(TF_IDF_array_query_binary, TF_IDF_array_binary)

# part b
print("<<<<<<<--------------- part b ------------------->>>>>>>")
print("compute 15 similar with BM25 model")
part_b(TF_IDF_array_query, TF_IDF_array)

# part c
print("<<<<<<<--------------- part c ------------------->>>>>>>")
print("evaluation metrics : p@5 p@10 MRR MAP")
print("compute metrics for result of part a : ")
name_a, result_a = parseResult(result_APath)
name_b, result_b = parseResult(result_BPath)
QID = [6,7,8,9,10]
judge = parseJudgment(judgmentPath)
print("analyse result of part A :")
part_c(name_a, result_a)
print("analyse result of part B :")
part_c(name_b,result_b)