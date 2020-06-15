"""

    test_model.py

    Tests for the GreynirTopic module

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

from greynir_topic import Model, Document, Corpus, Dictionary
from greynir_topic.model import CorpusIterator, w_from_lemma


@pytest.fixture(scope="module")
def model():
    """ Provide a module-scoped GreynirCorrect instance as a test fixture """
    g = Model("test")
    yield g
    # Do teardown here, if required


def test_basics():
    assert w_from_lemma("maður", "kk") == "maður/kk"
    assert w_from_lemma("að minnsta kosti", "ao") == "að_minnsta_kosti/ao"
    assert w_from_lemma("borgarstjórnar-fundur", "kk") == "borgarstjórnarfundur/kk"
    assert w_from_lemma("Vestur-Þýskaland", "hk") == "vesturþýskaland/hk"  # !!! FIXME
    assert w_from_lemma("dóms- og kirkjumálaráðherra", "kk") == "dóms_og_kirkjumálaráðherra/kk"  # !!! FIXME


class TestDocument(Document):

    def __init__(self, lemmas):
        self._lemmas = lemmas

    def __iter__(self):
        yield from self._lemmas


class TestCorpus(Corpus):

    @staticmethod
    def make_lemma(s):
        a = s.split("/")
        return a[0], a[1]

    @staticmethod
    def make_lemmas(s):
        return [TestCorpus.make_lemma(w) for w in s.split()]

    def __iter__(self):
        yield TestDocument(
            TestCorpus.make_lemmas("maður/kk fara/so út/ao í/fs búð/kvk")
        )
        yield TestDocument(
            TestCorpus.make_lemmas("búð/kvk vera/so lokaður/lo")
        )
        yield TestDocument(
            TestCorpus.make_lemmas("maður/kk vera/so leiður/lo")
        )
        yield TestDocument(
            TestCorpus.make_lemmas("hægur/lo vera/so að/nhm kaupa/so matur/kk í/fs búð/kvk")
        )


def test_dictionary():
    corpus = TestCorpus()
    ci = CorpusIterator(corpus)
    d = Dictionary(ci)
    assert "maður/kk" in d
    assert "búð/kvk" in d
    assert d.num_docs == 4


def test_init(model: Model):
    assert model._dimensions == Model._DEFAULT_DIMENSIONS


def test_train(model: Model):
    corpus = TestCorpus()
    model.train(corpus)
    s = ["maður/kk", "hundur/kk"]
    print(model.topic_vector(s))
