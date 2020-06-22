"""
    Greynir: Natural language processing for Icelandic

    Topic model subclasses using token streams

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

    This module subclasses the TupleModel class to support topic modeling
    for token streams, using plug-in lemmatization.

"""

from typing import Iterable, Union

from .tuplemodel import TupleDocument, LemmaTuple

from reynir import tokenize, TOK


StringIterable = Union[str, Iterable[str]]


class TokenDocument(TupleDocument):

    def __init__(self, text: StringIterable) -> None:
        super().__init__()
        self._text = text

    def gen_tuples(self) -> Iterable[LemmaTuple]:
        """ Generate (lemma, cat) tuples from the document """
        for t in tokenize(self._text):
            if t.kind == TOK.WORD:
                if t.val:
                    # Known word
                    yield (t.val[0].stofn, t.val[0].ordfl)
                else:
                    # Unknown word
                    yield (t.txt, "x")
            elif t.kind == TOK.PERSON:
                assert t.val
                # Name, gender
                yield (t.val[0][0], t.val[0][1])
            elif t.kind == TOK.ENTITY:
                # Entity name
                yield (t.txt, "entity")
            elif t.kind == TOK.COMPANY:
                # Company name
                yield (t.txt, "company")

