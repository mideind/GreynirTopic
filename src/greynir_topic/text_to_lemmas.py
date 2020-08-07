import tokenizer, os, subprocess, csv, json

class GenTokens():
    """
        This class converts a string to a tokenized string using the Greynir tokenizer
    """
    def __init__(self, text):
        self.text = text
    
    def get_tokens(self):
        token_list = list(tokenizer.tokenize(self.text))
        token_text = tokenizer.text_from_tokens(token_list)
        return token_text

class GenTaggedText():
    """
        This class takes in a tokenized text and relies on IceNLP to be installed as
        a subfolder in the programmers GreynirTopic directory.
        It then creates a text file containing the tokenized text and then tags it and 
        returns a subprocess.Popen object which contains the tagged text.
        This is temporary and will be replaced.
    """
    def __init__(self, tokenized_text):
        self.tokenized_text = tokenized_text
    
    def icetagger_tag(self):
        """ IceTagger is a temporary tagger. To be replaced by a more suitible tagger """
        os.chdir("IceNLP")
        os.chdir("bat")
        os.chdir("icetagger")
        with open("tokenized.txt", 'w') as f:
            f.write(self.tokenized_text)
        pipe = subprocess.Popen("cat 'tokenized.txt' | ./icetagger.sh -of 1", stdout=subprocess.PIPE,stderr=None, shell=True)
        for _ in range(3):
            os.chdir("..")
        return pipe

class GenLemmas():
    """ 
        This class uses Nefnir to lemmatize tagged text. It recieves tagged text and
        creates both a tagged text file and a lemmatized text file.
    """
    def __init__(self, tagged_text):
        self.tagged_text = tagged_text
    
    def nefnir_lemmatize(self):
        os.chdir("nefnir-master")
        with open("../tagged.txt", 'w') as f:
            f.write(self.tagged_text)
        subprocess.Popen("python3 nefnir.py -i ../tagged.txt -o ../lemmatized.txt -s ' ' ", shell = True).wait()
        os.chdir("..")

class ConvertText():
    """
        This class tokenizes, tags and lemmatizes a text file. As of now it relies on being
        initialized from the programmers GreynirTopic directory with the text file in the same
        directory and IceTagger and nefnir-master as subfolders but it will be simplified.
        It's also possible to get a json file from arnastofnun.is by calling create_as_json.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tok_text = GenTokens(open(self.filename)).get_tokens()
        self.tagged_text = GenTaggedText(self.tok_text)
    
    def lemmatize_text(self):
        t = self.tagged_text.icetagger_tag()
        fully_tagged_text = t.communicate()[0].decode('utf-8').replace(' <UNKNOWN>','')
        GenLemmas(fully_tagged_text).nefnir_lemmatize()
        with open('lemmatized.txt', 'r') as f:
            reader = csv.reader(f)
            all_rows = [row for row in reader]
        all_lemmas = list()
        for idx, row in enumerate(all_rows):
            if all_rows[idx]:
                if len(all_rows[idx][0].split()) == 3:
                    all_lemmas.append(all_rows[idx][0].split()[2])
        print(all_lemmas)
        print(len(all_lemmas))

    def create_as_json(self):
        """ Creates a json file containing tokens, lemmas and tags from arnastofnun.is """
        os.system("curl -X POST -F \"text=`cat "+self.filename+"`\" -F \"lemma=on\" -F \"expand_tag=on\" malvinnsla.arnastofnun.is > text_arnastofnun.json")
        with open('text_arnastofnun.json') as f:
            data = json.load(f)
        all_lemmas = list()
        for paragraph in data['paragraphs']:
            for sentence in paragraph['sentences']:
                for lemmas in sentence:
                    all_lemmas.append(lemmas['lemma'])
        print(all_lemmas)
        print(len(all_lemmas))

