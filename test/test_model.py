"""

    test_model.py

    Tests for the GreynirTopic Model and TupleModel modules

    Copyright (C) 2020 by Miðeind ehf.

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


    This module tests the topic vector builder functionality of GreynirTopic.

"""

import pytest

from greynir_topic import Model, Document, TokenDocument, Corpus, Dictionary
from greynir_topic.model import CorpusIterator
from greynir_topic.tuplemodel import w_from_lemma


@pytest.fixture(scope="module")
def model():
    """ Provide a module-scoped fresh Model instance as a test fixture """
    yield Model("test")


@pytest.fixture(scope="module")
def trained_model():
    """ Provide a module-scoped, trained Model instance as a test fixture """
    m = Model("test")
    test_train(m)
    yield m


def test_basics():
    assert w_from_lemma("maður", "kk") == "maður/kk"
    assert w_from_lemma("að minnsta kosti", "ao") == "að_minnsta_kosti/ao"
    assert w_from_lemma("borgarstjórnar-fundur", "kk") == "borgarstjórnarfundur/kk"
    assert w_from_lemma("Vestur-Þýskaland", "hk") == "vesturþýskaland/hk"  # !!! FIXME
    assert w_from_lemma("dóms- og kirkjumálaráðherra", "kk") == "dóms_og_kirkjumálaráðherra/kk"  # !!! FIXME


class DummyDocument(Document):

    def __init__(self, lemmas):
        self._lemmas = lemmas

    def __iter__(self):
        yield from self._lemmas


class DummyCorpus(Corpus):

    def __iter__(self):
        yield DummyDocument(
            "maður/kk fara/so út/ao í/fs búð/kvk".split()
        )
        yield DummyDocument(
            "búð/kvk vera/so lokaður/lo".split()
        )
        yield DummyDocument(
            "maður/kk vera/so leiður/lo".split()
        )
        yield DummyDocument(
            "hægur/lo vera/so að/nhm kaupa/so matur/kk í/fs búð/kvk".split()
        )


def test_dictionary():
    corpus = DummyCorpus()
    ci = CorpusIterator(corpus)
    d = Dictionary(ci)
    assert "maður/kk" in d
    assert "búð/kvk" in d
    assert d.num_docs == 4


def test_init(model: Model):
    assert model._dimensions == Model._DEFAULT_DIMENSIONS


def test_train(model: Model):
    corpus = DummyCorpus()
    # For testing purposes, we include all lemmas in the dictionary,
    # even the rare ones (which would be omitted by default in normal processing)
    model.train(corpus, min_count=0)
    s = ["maður/kk", "hundur/kk"]
    tv = model.topic_vector(s)
    s2 = ["maður/kk", "búð/kvk"]
    tv2 = model.topic_vector(s2)
    tv3 = model.topic_vector(s2)
    assert model.similarity(tv, tv2) > 0.90
    assert model.similarity(tv2, tv3) > 0.9999


def test_load(trained_model: Model):
    """ Test loading an existing model - assuming it's
        already been trained in test_train() """
    # Should not need to retrain the model
    s = ["maður/kk", "hundur/kk"]
    tv = trained_model.topic_vector(s)
    s2 = ["maður/kk", "búð/kvk"]
    tv2 = trained_model.topic_vector(s2)
    tv3 = trained_model.topic_vector(s2)
    assert trained_model.similarity(tv, tv2) > 0.90
    assert trained_model.similarity(tv2, tv3) > 0.9999


class TokenCorpus(Corpus):

    def __iter__(self):
        yield TokenDocument(
            "Maður fór út í búð."
        )
        yield TokenDocument(
            "Búðin var lokuð."
        )
        yield TokenDocument(
            "Maðurinn varð leiður."
        )
        yield TokenDocument(
            "Hægt er að kaupa mat í búðum."
        )


def test_train_token(model: Model):
    corpus = TokenCorpus()
    # For testing purposes, we include all lemmas in the dictionary,
    # even the rare ones (which would be omitted by default in normal processing)
    model.train(corpus, min_count=0)
    s = ["maður/kk", "hundur/kk"]
    tv = model.topic_vector(s)
    s2 = ["maður/kk", "búð/kvk"]
    tv2 = model.topic_vector(s2)
    tv3 = model.topic_vector(s2)
    assert model.similarity(tv, tv2) > 0.90
    assert model.similarity(tv2, tv3) > 0.9999

