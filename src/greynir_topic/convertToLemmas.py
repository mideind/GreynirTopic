import tokenizer, os, subprocess, csv

class gen_Tokens():
    """
        This class converts a string to a tokenized string using the Greynir tokenizer
    """
    def __init__(self, text):
        self.text = text
    
    def get_Tokens(self):
        tokenList = list(tokenizer.tokenize(self.text))
        tokenText = tokenizer.text_from_tokens(tokenList)
        return tokenText

class gen_TaggedText():
    """
        This class takes in a tokenized text and relies on IceNLP to be installed as
        a subfolder in the programmers GreynirTopic directory.
        It then creates a text file containing the tokenized text and then tags it and 
        returns a subprocess.Popen object which contains the tagged text.
        This is temporary and will be replaced.
    """
    def __init__(self, tokenized_Text):
        self.tokenized_Text = tokenized_Text
    
    def iceTagger_tag(self):
        """ IceTagger is a temporary tagger. To be replaced by a more suitible tagger """
        os.chdir("IceNLP")
        os.chdir("bat")
        os.chdir("icetagger")
        text = open("tokenized.txt", 'w')
        text.write(self.tokenized_Text)
        pipe = subprocess.Popen("cat 'tokenized.txt' | ./icetagger.sh -of 1", stdout=subprocess.PIPE,stderr=None, shell=True)
        for _ in range(3):
            os.chdir("..")
        return pipe

class gen_Lemmas():
    """ 
        This class uses Nefnir to lemmatize tagged text. It recieves tagged text and
        creates both a tagged text file and a lemmatized text file.
    """
    def __init__(self, tagged_Text):
        self.tagged_text = tagged_Text
    
    def nefnir_Lemmatize(self):
        os.chdir("nefnir-master")
        text = open("../tagged.txt", 'w')
        text.write(self.tagged_text)
        text.close()
        subprocess.Popen("python3 nefnir.py -i ../tagged.txt -o ../lemmatized.txt -s ' ' ", shell = True).wait()
        os.chdir("..")

class convert_Text():
    """
        This class tokenizes, tags and lemmatizes a text file. As of now it relies on being
        initialized from the programmers GreynirTopic directory with the text file in the same
        directory and IceTagger and nefnir-master as subfolders but it will be simplified.
        It's also possible to get a json file from arnastofnun.is by calling create_AS_json.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tokText = gen_Tokens(open(self.filename)).get_Tokens()
        self.taggedText = gen_TaggedText(self.tokText)
    
    def lemmatize_Text(self):
        t = self.taggedText.iceTagger_tag()
        fully_tagged_text = t.communicate()[0].decode('utf-8').replace(' <UNKNOWN>','')
        gen_Lemmas(fully_tagged_text).nefnir_Lemmatize()
        lemmas = open('lemmatized.txt', 'r')
        reader = csv.reader(lemmas)
        allRows = [row for row in reader]
        allLemmas = []
        for idx, row in enumerate(allRows):
            if allRows[idx]:
                if len(allRows[idx][0].split()) == 3:
                    allLemmas.append(allRows[idx][0].split()[2])
        lemmas.close()
        print(allLemmas)

    def create_AS_json(self):
        """ Creates a json file containing tokens, lemmas and tags from arnastofnun.is """
        os.system("curl -X POST -F \"text=`cat "+self.filename+"`\" -F \"lemma=on\" -F \"expand_tag=on\" malvinnsla.arnastofnun.is > tokens_arnastofnun.json")
    
    

