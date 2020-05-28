! pip install git+https://gitlab.fi.muni.cz/xstefan3/pv211-utils.git@master | grep '^Successfully'
! pip install pyspellchecker

import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from nltk.corpus import stopwords
from collections import Counter

def tokenization(sentence):
    """Return the tokenized sentence."""
    tokens = nltk.word_tokenize(sentence)
    return " ".join(tokens)
	
def lowercase(sentence):
    """Return the sentence with its letters transformed into lowercase."""
    return sentence.lower()

def removePunctuation(sentence):
    """Return the sentence with the symbols below removed."""
    symbols = "!\"#$%&()*+-/:,;'<=>?@[\]^_`{|}~\n"

    sentence = np.char.replace(sentence, ".", '') 
    for i in symbols:
        sentence = np.char.replace(sentence, i, ' ')
    return sentence.item()
	
def removeSingleCharacters(sentence):
    """Return the sentence with all the characters with the length of 
    1 removed."""
    words = sentence.split()
    sentence = ''

    for word in words:
        if len(word) > 1:
            sentence = sentence + " " + word
      
    return sentence
	
def removeStopWords(sentence):
    """Return the sentence with its stop words removed."""
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    sentence = ''

    for word in words:
        if word not in stop_words:
            sentence = sentence + " " + word
        
    return sentence
	
def lemmatization(sentence):
    """Return the sentence with its words lemmatized."""
    lemmatizer = WordNetLemmatizer()
    words = sentence.split()
    sentence = []

    for word in words:
        sentence.append(lemmatizer.lemmatize(word))
        sentence.append(" ")
    
    return "".join(sentence) 
	
def stemming(sentence):
    """Return the sentence with its words stemmed."""
    words = sentence.split()
    stemmer = PorterStemmer()
    sentence = []

    for word in words:
        sentence.append(stemmer.stem(word))
        sentence.append(" ")
    return "".join(sentence)  
	
def preprocess(sentence):
    """Preprocess the given sentence."""
    sentence = tokenization(sentence)
    sentence = lowercase(sentence)
    sentence = removePunctuation(sentence)
    sentence = lemmatization(sentence)
    sentence = stemming(sentence)
    sentence = removeSingleCharacters(sentence)
    sentence = removeStopWords(sentence)
    return sentence
	
from pv211_utils.entities import DocumentBase

class Document(DocumentBase):
    def __init__(self, document_id, authors, bibliography, title, body):
        super().__init__(document_id, authors, bibliography, title, body)
        self.authors = preprocess(self.authors)
        self.bibliography = preprocess(self.bibliography)
        self.title = preprocess(self.title)
        self.body = preprocess(self.body)

from pv211_utils.loader import load_documents
documents = load_documents(Document)  

from pv211_utils.entities import QueryBase

class Query(QueryBase):
    def __init__(self, query_id, body):
        super().__init__(query_id, body)
        self.body = preprocess(self.body)

from pv211_utils.loader import load_queries
queries = load_queries(Query)

from pv211_utils.loader import load_judgements
relevant = load_judgements(queries, documents)

def addTokens(sentence, i, dictionary):
    """Add the tokens of the sentence to the given dictionary and set the index
    of the document its value.

    Parameters
    ----------
    sentence : str
        A sentence which tokens will be set as the keys of the dictionary unless
        they do not already exist.
    i : int
        An index of the document that contains the sentence and the tokens.
    dictionary : dict of (token, i)
        A dictionary that has a token as a key and a index of the document that
        contains the token as its value.

    Return
    ------
    dict of (token, i)
        The given dictionary with the added keys and values.

    """
    tokens = sentence.split()
    for token in tokens:
        try:
            dictionary[token].add(i)
        except:
            dictionary[token] = {i}  
    return dictionary
	
def docFrequency(documents):
    """Count the frequencies of all tokens within the collection of documents 
    and set them into dictionary. 

    Parameters
    ----------
    documents : OrderedDict
        A collection of documents.

    Return
    ------
    dict of (token, frequency)
        A dictionary with tokens in its keys and their frequencies within the 
        documents as values.

    """
    df = {}

    for i in range(1,len(documents)+1):
        df = addTokens(documents[i].body, i, df)
        df = addTokens(documents[i].title, i, df)
        df = addTokens(documents[i].authors, i, df)
        df = addTokens(documents[i].bibliography, i, df)

    for token in df:
        df[token] = len(df[token])

    return df

DF = docFrequency(documents)

def wordFrequency(word):
    """Return the frequency of a given word."""
    try:
        frequency = DF[word]
    except:
        frequency = 0
    return frequency
	
def countTfidf(tokens, i, counter, dictionary):
    """Determine the tfidf weight for each token within the document at index i
    and add it to the dictionary.

    Parameters
    ----------
    tokens : list
        A list of tokens.
    i : int
        An index of the document that contains the tokens.
    counter : counter
        A counter that counts the frequencies of tokens.
    dictionary : dict of (index, token, tfidf weight)
        A dictionary that has the index and the token as its keys and the 
        corresponding tdidf weight as their values.

    Return
    ------
    dict of (index, token, tfidf weight)
        An updated dictionary.

    """
    for token in np.unique(tokens):
        tf = (1+np.log(counter[token]))
        df = wordFrequency(token)
        idf = np.log((len(documents))/(df))
        dictionary[i, token] = tf*idf  
    return dictionary
	
def tfidf(documents):
    """Determine the complete dictionary of all the tokens within the documents 
    and their corresponding tfidf weights.

    Parameters
    ----------
    documents : OrderedDict
        A collection of documents. 

    Return
    ------
    dict of (index, token, tfidf weight)
        A complete dictionary of tfidf weights of the tokens.

    """
    tfidf = {}
    tfidf_title = {}
    tfidf_authors = {}
    tfidf_bibliography = {}

    for i in range(1, len(documents)+1):
        body = documents[i].body.split()
        counter = Counter(body + documents[i].title.split() + 
                          documents[i].authors.split() + 
                          documents[i].bibliography.split())
        tfidf = countTfidf(body, i, counter, tfidf)

        title = documents[i].title.split()
        counter = Counter(title + documents[i].body.split() + 
                          documents[i].authors.split() + 
                          documents[i].bibliography.split()) 
        tfidf_title = countTfidf(title, i, counter, tfidf_title)

        for token in tfidf_title:
            tfidf[token] = tfidf_title[token]

        authors = documents[i].authors.split()
        counter = Counter(authors + documents[i].body.split() + 
                          documents[i].title.split() + 
                          documents[i].bibliography.split())
        tfidf_authors = countTfidf(authors, i, counter, tfidf_authors)

        for token in tfidf_authors:
            tfidf[token] = tfidf_authors[token]

        bibliography = documents[i].bibliography.split()
        counter = Counter(bibliography + documents[i].body.split() + 
                          documents[i].title.split() + 
                          documents[i].authors.split())
        tfidf_bibliography = countTfidf(bibliography, i, counter, tfidf_bibliography)

        for token in tfidf_bibliography:
            tfidf[token] = tfidf_bibliography[token]
    
    return tfidf

TFIDF = tfidf(documents)

def queryTokens(sentence):
    """Determine a dictionary of all the tokens in the query sentence and their
    corresponding idf (informativeness) weights.

    Parameters
    ----------
    sentence : str
        A query sentence.

    Return
    ------
    dict of (token, idf weight)
        A dictionary with each token of the query as its keys and the 
        corresponding idf weights as their values.

    """
    querytokens = {}
    tokens = preprocess(sentence).split()

    for token in tokens:
        df = wordFrequency(token)
        if df == 0:
            idf = 0
        else:
            idf = np.log((len(documents))/(df))
        if token in querytokens:
            querytokens[token] += idf
        else:
            querytokens[token] = idf

    norm = 0

    for token in querytokens:
        norm += (querytokens[token]*querytokens[token])
    norm = np.sqrt(norm)

    for token in querytokens:
        querytokens[token] = querytokens[token]/norm

    return querytokens
	
def documentTokens(document):
    """Determine a dictionary of all the tokens in the document and their
    corresponding tfidf weights. To increase precision, weights from different 
    zones such as title and authors are scaled as they are not as important 
    as the body zone. 

    Parameters
    ----------
    document : doc
        A document that contains a body, a title, authors and a bibliography.

    Return
    ------
    dict of (word, tfidf weight)
        A dictionary that has the tokens of the given document as its keys and the
        corresponding tfidf weights as their values.

    """
    documenttokens = {}
    norm = 0  

    body = document.body.split()
    title = document.title.split()
    authors = document.authors.split()
    bibliography = document.bibliography.split()
    words = body + title + authors + bibliography  

    for word in np.unique(words):
        norm += (TFIDF[(document.document_id,word)]*
                 TFIDF[(document.document_id,word)])
    norm = np.sqrt(norm)

    for word in np.unique(body):
        documenttokens[word] = (TFIDF[(document.document_id,word)]/norm)
    for word in np.unique(title):
        documenttokens[word] = (TFIDF[(document.document_id,word)]/norm)*1.17
    for word in np.unique(authors):
        documenttokens[word] = (TFIDF[(document.document_id,word)]/norm)*1.6
    for word in np.unique(bibliography):
        documenttokens[word] = (TFIDF[(document.document_id,word)]/norm)*1.31

    return documenttokens
	
from pv211_utils.irsystem import IRSystem

class IRSystem(IRSystem):

    def __init__(self):
        self.documents = list(documents.values())

    def search(self, query):
        """Rank the documents in a descending order of relevance to a given 
        query. The dictionaries of the query and the document are treated as 
        vectors so that the similarity is measured by the cosine similarity of 
        the weights in the dictionaries.

        Parameters
        ----------
        query : str
            A query sentence.

        Return
        ------
        list
            A final result of the search process that contains the ordered list 
            of documents.

        """
        querytokens = queryTokens(query.body)
        ranking = []

        for document in self.documents:
            documenttokens = documentTokens(document)
            similarity = 0

            for querytoken in querytokens:
                if querytoken in documenttokens:
                    similarity = similarity + (querytokens[querytoken]*
                                               documenttokens[querytoken])

            ranking.append((document,similarity))

        ranking = sorted(ranking, key=lambda tup: tup[1], reverse=True) 
        result = []
    
        for x in ranking:
            result.append(x[0])

        return result
		
from pv211_utils.eval import mean_average_precision

mean_average_precision(IRSystem(), submit_result=True, author_name="Kuikka, Toni")