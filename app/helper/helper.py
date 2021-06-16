import spacy
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors

# instanciando nlp
nlp = spacy.load("pt_core_news_sm", 
                    disable=["paser", 
                             "ner", 
                             "tagger", 
                             "textcat"])

# isntanciando uma corpora
w2v_modelo_cbow = KeyedVectors.load_word2vec_format("./corpora/cbow_s300.txt")

def tokenizador(texto):
    """
        recebe um texto e tokeniza, lowe_case, retira numeros e stop words
    """
    doc = nlp(texto)
    tokens_validos = []
    for token in doc:
        e_valido = not token.is_stop and token.is_alpha
        if e_valido:
            tokens_validos.append(token.text.lower())

    
    return  tokens_validos


def combinacao_de_vetores(palavras, modelo):
    """
        recebe palavras tokenizadas e um modelo para transformar em um vetor
    """
    vetor_resultante = np.zeros((1,300))

    for pn in palavras:
        try:
            vetor_resultante += modelo.get_vector(pn)

        except KeyError:
            pass
                

    return vetor_resultante

def matriz_vetores(textos, modelo):
    """
        utilizando o combinação de textos para várias linhas do dataset
    """
    x = len(textos)
    y = 300
    matriz = np.zeros((x,y))

    for i in range(x):
        palavras = tokenizador(textos.iloc[i])
        matriz[i] = combinacao_de_vetores(palavras, modelo)

    return matriz

def classificador(modelo, x_treino, y_treino):
    """
        Treinando um Classificador
    """
    RL = LogisticRegression(max_iter = 800)
    RL.fit(x_treino, y_treino)
    
    return RL

def predict_model(sentence, model_name):
    """
        Recebe uma sentença e uma instância de modelo e retorna uma classificação
    """
    modelo_carregado = pickle.load(
                                    open(
                                        f'./models/{model_name}.sav','rb'
                                        ))
    frase_tokenizada = tokenizador(sentence)
    frase_tokenizada_vetorizada = combinacao_de_vetores(frase_tokenizada, w2v_modelo_cbow)
    frase_classificada = modelo_carregado.predict(frase_tokenizada_vetorizada)
    
    return frase_classificada[0]