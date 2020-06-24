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

from typing import Iterable, Union, Optional

from .tuplemodel import TupleDocument, LemmaTuple

from reynir import Greynir, tokenize, paragraphs, Tok, TOK


# The type of the object that can be passed to TokenDocument for tokenization
StringIterable = Union[str, Iterable[str]]


class TokenDocument(TupleDocument):

    """ This class consumes a string or an iterable of strings,
        producing a stream of (lemma, category) tuples via the overridable
        function lemmatize(). The default implementation of lemmatize()
        is very simplistic but may be sufficient for basic use cases. """

    def __init__(self, text_or_gen: StringIterable) -> None:
        super().__init__()
        self._text = text_or_gen

    def gen_tuples(self) -> Iterable[LemmaTuple]:
        """ Generate (lemma, cat) tuples from the document by
            tokenizing it into paragraphs and sentences, then
            lemmatizing each sentence """
        tokens = tokenize(self._text)
        for pg in paragraphs(tokens):
            for _, sent in pg:
                yield from self.lemmatize(sent)

    def lemmatize(self, sent: Iterable[Tok]) -> Iterable[LemmaTuple]:
        """ Lemmatize a sentence (list of tokens), returning
            an iterable of (lemma, category) tuples """
        # Super-simplistic lemmatizer: return the first lemma
        # from the list of possible lemmas of a word
        for t in sent:
            if t.kind == TOK.WORD:
                if t.val:
                    # Known word
                    yield (t.val[0].stofn, t.val[0].ordfl)
                else:
                    # Unknown word: assume it's an entity
                    yield (t.txt, "entity")
            elif t.kind == TOK.PERSON:
                assert t.val
                # Name, gender
                yield (t.val[0][0], "person_" + t.val[0][1])
            elif t.kind == TOK.ENTITY or t.kind == TOK.COMPANY:
                # Entity or company name
                yield (t.txt, "entity")


class ParsedDocument(TokenDocument):

    """ This subclass of TokenDocument uses the Greynir parser to
        lemmatize sentences, falling back to the TokenDocument lemmatizer
        for sentences that can't be parsed. """

    _g = None  # type: Optional[Greynir]

    def __init__(self, text_or_gen: StringIterable) -> None:
        super().__init__(text_or_gen)

    def lemmatize(self, sent: Iterable[Tok]) -> Iterable[LemmaTuple]:
        """ Lemmatize a sentence (list of tokens), returning
            an iterable of (lemma, category) tuples """
        if self._g is None:
            # Initialize parser singleton
            self.__class__._g = Greynir()
        # Attempt to parse the sentence
        assert self._g is not None
        s = self._g.parse_tokens(sent)
        if s.tree is None:
            # Unable to parse: fall back to simple lemmatizer
            yield from super().lemmatize(sent)
        else:
            # Successfully parsed: obtain the (lemma, category) tuples
            # from the terminals of the parse tree
            yield from s.tree.lemmas_and_cats
