import re

class DataCleaner():

    @classmethod
    def lowercase(cls, text):
        return text.lower()
    
    @classmethod
    def replace_spaces(cls, text):
        text = re.sub(r"\'s", " ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        return text

    @classmethod
    def remove_contractions(cls, text):   
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        return text

    @classmethod
    def remove_html(cls,text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(text))
        return cleantext
    
    @classmethod
    def remove_punctuation(cls,text):
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',text)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned

    @classmethod
    def clean_text_for_classical_methods(cls,text):
        text = cls.remove_html(text)
        text = cls.lowercase(text)
        text = cls.replace_spaces(text)
        text = cls.remove_contractions(text)
        text = cls.remove_punctuation(text)
        
        return text
