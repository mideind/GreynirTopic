"""
    Greynir: Natural language processing for Icelandic

    Topic model class definition

    Copyright (C) 2020 Miðeind ehf.
    Original author: Vilhjálmur Þorsteinsson

    This software is licensed under the MIT License:

        Permission is hereby granted, free of charge, to any person
        obtaining a copy of this software and associated documentation
        files (the "Software"), to deal in the Software without restriction,
        including without limitation the rights to use, copy, modify, merge,
        publish, distribute, sublicense, and/or sell copies of the Software,
        and to permit persons to whom the Software is furnished to do so,
        subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    This module is written in Python 3

    This module reads documents as bags-of-words
    and indexes them using Latent Semantic Indexing (LSI, also called Latent Semantic
    Analysis, LSA), with indexes generated with the help of the Gensim document
    processing module.

    The indexing proceeds in stages (cf. https://radimrehurek.com/gensim/tut2.html):

    1) Conversion of document contents
        into a corpus stream, yielding each document as a bag-of-words
        via the CorpusIterator class. The corpus stream is
        filtered so that it only contains significant verbs,
        nouns, adjectives and person and entity names - all normalized
        (i.e. verbs to 'nafnháttur', nouns to nominative singular, and
        adjectives to normal nominative singular masculine).

    2) Generation of a Gensim dictionary (vocabulary) across the corpus stream,
        cutting out rare words, resulting in a word count vector.

    3) Calculation of word weights from the dictionary via the TFIDF algorithm,
        generating a TFIDF vector (TFIDF=term frequency–inverse document frequency,
        cf. http://www.tfidf.com/).

    4) Generation of the LSI lower-dimensionality model (matrix) from the corpus
        after transformation of each document through the TFIDF vector.

    After the LSI model has been generated, it can be used to calculate LSI
    vectors for any set of words. Subsequently, the closeness of any document
    to a topic can be estimated by calculating the cosine similarity between
    the document's LSI vector and the topic's LSI vector.

"""

from typing import Iterator, Tuple, List, Union, Optional

import os
import sys
import time
from abc import ABC, abstractmethod

from gensim import corpora, models, matutils  # type: ignore


# A TopicVector is a sparse array of floats,
# i.e. a list of (index, content) tuples
TopicVector = List[Tuple[int, float]]
# A LemmaTuple contains two strings, the lemma and its category
LemmaTuple = Tuple[str, str]
# A LemmaString contains a lemma and its category, separated by a slash '/'
LemmaString = str


def w_from_lemma(lemma: str, cat: str) -> LemmaString:
    """ Convert a (lemma, cat) tuple to a bag-of-words key """
    return lemma.lower().replace("-", "").replace(" ", "_") + "/" + cat


class Document(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[LemmaTuple]:
        """ Yield a stream of (lemma, cat) tuples from the document """
        ...


class Corpus(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Document]:
        """ Yield a stream of Document objects """
        ...


class Dictionary(corpora.Dictionary):

    """ Subclass of gensim.corpora.Dictionary that adds a __contains__
        operator for easy membership check """

    def __init__(self, iterator: "CorpusIterator") -> None:
        super().__init__(iterator)

    def __contains__(self, word: str) -> bool:
        return word in self.token2id


class CorpusIterator:

    """ Iterate through a collection of documents,
        yielding a list of "lemma/cat" strings
        or a bag-of-words for each document """

    def __init__(self, corpus: Corpus, dictionary: Dictionary=None):
        self._corpus = corpus
        self._dictionary = dictionary
        if self._dictionary is not None:
            # If this iterator is associated with a dictionary, use it to
            # return bags-of-words using dictionary indices
            self._xform = lambda x: self._dictionary.doc2bow(x)
        else:
            # No dictionary: return the lemma/cat strings as-is
            self._xform = lambda x: x

    def __iter__(self) -> Iterator[Union[List[LemmaString], List[int]]]:
        """ Iterate through documents and return a lemma/cat list or
            a bag of words for each of them """
        xform = self._xform
        for document in self._corpus:
            lemmas = [w_from_lemma(lemma, cat) for lemma, cat in document]
            if lemmas:
                yield xform(lemmas)


class Model:

    """ Wraps the document indexing functionality """

    # Default number of dimensions in topic vectors
    _DEFAULT_DIMENSIONS = 200

    # The default directory for model data files is the ./models directory
    _DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

    def __init__(self, name: str, *,
        directory: str=None,
        dimensions: int=None,
        min_count: int=3
    ) -> None:
        """ Create a model instance.
            name: the name of the model, included in data file names.
            directory: the directory where data files will be written.
            dimensions: the topic vector dimensions, typically 200.
            min_count: the minimum number of occurrences of a lemma
                for it to be included in the model dictionary.
        """
        self._name = name
        self._dimensions = dimensions or self._DEFAULT_DIMENSIONS
        if directory:
            # Override class-wide default for this model instance
            self._DIRECTORY = directory
        self._min_count = min_count
        self._dictionary = None  # type: Optional[Dictionary]
        self._tfidf = None
        self._model = None

    def _filename_from_ext(self, ext: str) -> str:
        """ Return a full file path from a given extension """
        return os.path.join(self._DIRECTORY, self._name + "." + ext)

    @property
    def dictionary_filename(self) -> str:
        return self._filename_from_ext("dict")

    @property
    def plain_corpus_filename(self) -> str:
        return self._filename_from_ext("corpus.mm")

    @property
    def tfidf_corpus_filename(self) -> str:
        return self._filename_from_ext("corpus-tfidf.mm")

    @property
    def tfidf_model_filename(self) -> str:
        return self._filename_from_ext("tfidf.model")

    @property
    def lsi_model_filename(self) -> str:
        return self._filename_from_ext("lsi.model")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def train_dictionary(self, corpus_iterator: CorpusIterator) -> None:
        """ Iterate through the document corpus
            and create a fresh Gensim dictionary """
        dic = Dictionary(corpus_iterator)
        # Drop words that only occur very few times in the entire set
        if self._min_count > 0:
            dic.filter_extremes(no_below=self._min_count, keep_n=None)
            # !!! TODO: We may want to use other Gensim filtering
            # !!! features, such as dropping lemmas that occur in
            # !!! almost all documents
        # We must have something in our dictionary
        assert len(dic.token2id) > 0
        dic.save(self.dictionary_filename)
        self._dictionary = dic

    def load_dictionary(self) -> None:
        """ Load a dictionary from a previously prepared file """
        self._dictionary = Dictionary.load(self.dictionary_filename)

    def train_plain_corpus(self, corpus_iterator: CorpusIterator) -> None:
        """ Create a plain vector corpus, where each vector represents a
            document. Each element of the vector contains the count of
            the corresponding word (as indexed by the dictionary) in
            the document. """
        corpora.MmCorpus.serialize(self.plain_corpus_filename, corpus_iterator)

    def load_plain_corpus(self) -> corpora.MmCorpus:
        """ Load the plain corpus from file """
        return corpora.MmCorpus(self.plain_corpus_filename)

    def train_tfidf_model(self) -> None:
        """ Create a fresh TFIDF model from a dictionary """
        if self._dictionary is None:
            self.load_dictionary()
        tfidf = models.TfidfModel(dictionary=self._dictionary)
        tfidf.save(self.tfidf_model_filename)
        self._tfidf = tfidf

    def load_tfidf_model(self) -> None:
        """ Load an already generated TFIDF model """
        self._tfidf = models.TfidfModel.load(self.tfidf_model_filename, mmap="r")

    def train_tfidf_corpus(self) -> None:
        """ Create a TFIDF corpus from a plain vector corpus """
        if self._tfidf is None:
            self.load_tfidf_model()
        assert self._tfidf is not None
        corpus = self.load_plain_corpus()
        corpus_tfidf = self._tfidf[corpus]
        corpora.MmCorpus.serialize(self.tfidf_corpus_filename, corpus_tfidf)

    def load_tfidf_corpus(self) -> corpora.MmCorpus:
        """ Load a TFIDF corpus from file """
        return corpora.MmCorpus(self.tfidf_corpus_filename)

    def train_lsi_model(self, **kwargs) -> None:
        """ Train an LSI model from the entire document corpus """
        corpus_tfidf = self.load_tfidf_corpus()
        if self._dictionary is None:
            self.load_dictionary()
        # Initialize an LSI transformation
        lsi = models.LsiModel(
            corpus_tfidf,
            id2word=self._dictionary,
            num_topics=self._dimensions,
            **kwargs
        )
        # Save the generated model
        lsi.save(self.lsi_model_filename)

    def load_lsi_model(self) -> None:
        """ Load a previously generated LSI model """
        self._model = models.LsiModel.load(self.lsi_model_filename, mmap="r")

    def train(self, corpus: Corpus, *, keep_temp_files: bool=False) -> None:
        """ Go through all training steps for a document corpus,
            ending with an LSI model built on TF-IDF vectors
            for each document """
        self.train_dictionary(
            CorpusIterator(corpus, dictionary=None)
        )
        self.train_plain_corpus(
            CorpusIterator(corpus, dictionary=self._dictionary)
        )
        self.train_tfidf_model()
        self.train_tfidf_corpus()
        self.train_lsi_model()
        if not keep_temp_files:
            # Remove intermediate model files that are only
            # used during training, not during inference
            os.remove(self.plain_corpus_filename)
            os.remove(self.tfidf_corpus_filename)

    def topic_vector(
        self, lemmas: Union[List[LemmaTuple], List[LemmaString]]
    ) -> TopicVector:
        """ Return a sparse topic vector for a list of lemmas,
            which can contain either "lemma/category" strings or
            ("lemma", "category") tuples. """
        if not lemmas:
            return []
        if isinstance(lemmas[0], tuple):
            lemmas = [w_from_lemma(lemma, cat) for lemma, cat in lemmas]
        else:
            assert all("/" in lemma for lemma in lemmas)  # Must contain a slash
        if self._dictionary is None:
            self.load_dictionary()
        assert self._dictionary is not None
        if self._tfidf is None:
            self.load_tfidf_model()
        assert self._tfidf is not None
        if self._model is None:
            self.load_lsi_model()
        assert self._model is not None
        bag = self._dictionary.doc2bow(lemmas)
        if not bag:
            return []
        tfidf = self._tfidf[bag]
        return self._model[tfidf]

    @staticmethod
    def similarity(topic_vector_a: TopicVector, topic_vector_b: TopicVector) -> float:
        """ Return the cosine similarity between two sparse topic vectors """
        return matutils.cossim(topic_vector_a, topic_vector_b)
