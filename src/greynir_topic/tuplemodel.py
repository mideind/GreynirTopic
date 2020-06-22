"""
    Greynir: Natural language processing for Icelandic

    Topic model subclasses using (lemma, category) tuples

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

    This module subclasses the Document and Model classes from model.py
    to implement a (lemma, category) tuple-based interface on the topic
    modeling functionality.

"""

from typing import Iterator, Iterable, Tuple, List, Union, cast

from abc import abstractmethod

from .model import Model, Document, TopicVector, LemmaString


# A LemmaTuple contains two strings, the lemma and its category
LemmaTuple = Tuple[str, str]


def w_from_lemma(lemma: str, cat: str) -> LemmaString:
    """ Convert a (lemma, cat) tuple to a bag-of-words key """
    return lemma.lower().replace("-", "").replace(" ", "_") + "/" + cat


class TupleDocument(Document):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def gen_tuples(self) -> Iterable[LemmaTuple]:
        ...

    def __iter__(self) -> Iterator[LemmaString]:
        """ Yield a stream of lemmas from the document """
        for lemma, cat in self.gen_tuples():
            yield w_from_lemma(lemma, cat)


class TupleModel(Model):

    """ Wraps the topic vector modeling functionality """

    def topic_vector(
        self, lemmas: Union[List[LemmaTuple], List[LemmaString]]
    ) -> TopicVector:
        """ Return a sparse topic vector for a list of lemmas,
            which can contain either "lemma/category" strings or
            ("lemma", "category") tuples. """
        if not lemmas:
            return []
        if isinstance(lemmas[0], tuple):
            lemmas = cast(List[LemmaTuple], lemmas)
            lemmas = [w_from_lemma(lemma, cat) for lemma, cat in lemmas]
        else:
            assert all("/" in lemma for lemma in lemmas)  # Must contain a slash
        return super().topic_vector(cast(List[LemmaString], lemmas))
