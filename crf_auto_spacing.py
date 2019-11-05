from __future__ import unicode_literals, print_function

import re

raw_train = 'E:\Programming\python\창회선배스터디\창회선배 자료\토지2.txt'
raw_test = 'E:\Programming\python\창회선배스터디\창회선배 자료\Test-earthquake.txt'

def r2c(raw_path, corpus_path):
    raw = open(raw_path, encoding='utf-8')
    raw_sentences = raw.read().split('\n') #문장단위 토큰 나누기
    corpus = open(corpus_path, 'w', encoding='utf-8')
    sentences = []
    for raw_sentences in raw_sentences:
        if not raw_sentences:
            continue
        text = re.sub(r'(\ )+',' ',raw_sentences).strip()
        taggeds =[]
        for i in range(len(text)):
            if i == 0:
                taggeds.append('{}/B'.format(text[i])) #시작부분 'B'태그부여/ .format(ㅁ)은 {}안에 ㅁ이들어간다
            elif text[i] != ' ':
                successor = text[i - 1]
                if successor == ' ':
                    taggeds.append('{}/B'.format(text[i])) #띄어쓰기 다음 태그부여
                else:
                    taggeds.append('{}/I'.format(text[i])) #문자 다음 태그부여
        sentences.append(' '.join(taggeds))
    corpus.write('\n'.join(sentences))

r2c(raw_train,'train.txt')
r2c(raw_test, 'test.txt')

def c2s(path): #train.txt를 리스트 형식으로 전환하는 함수
    corpus = open(path, encoding='utf-8').read()
    raws = corpus.split('\n')
    sentences=[]
    for raw in raws:
        tokens = raw.split(' ') #공백기준 토큰 분류
        sentence = []
        for token in tokens:
            try:
                word, tag = token.split('/')
                if word and tag:
                    sentence.append([word,tag])
            except:
                pass
        sentences.append(sentence)
    return sentences

sent0 = c2s('train.txt')
print(sent0[:20])

def i2f(sent, i, offset):
    word, tag = sent[i + offset]
    if offset < 0:
        sign = ''
    else:
        sign = '+'
    return '{}{}:word={}'.format(sign, offset, word)

def w2f(sent, i): #좌우 i 글자만큼 보고 인덱스의 글자에 태그를 맞추기 위한 함수
    L = len(sent)
    word, tag = sent[i]
    features = ['bias']
    features.append(i2f(sent, i, 0))
    if i > 1:
        features.append(i2f(sent, i, -2)) #좌우 길이 숫자 바꿔보기
    if i > 0:
        features.append(i2f(sent, i, -1))
    else:
        features.append('bos')
    if i < L - 2:
        features.append(i2f(sent, i, 2))
    if i < L - 1:
        features.append(i2f(sent, i, 1))
    else:
        features.append('eos')
    return features

def s2w(sent):
    return [word for word, tag in sent]

def s2t(sent):
    return [tag for word, tag in sent]

def s2f(sent):
    return [w2f(sent, i) for i in range(len(sent))]

########학습#########

import pycrfsuite

train_sents = c2s('train.txt') #리스트로 전환
test_sents = c2s('test.txt')  #리스트로 전환
train_x = [s2f(sent) for sent in train_sents] #feature을 추출한 sentensces
train_y = [s2f(sent) for sent in train_sents]
test_x = [s2f(sent) for sent in test_sents]
test_y = [s2f(sent) for sent in test_sents]
trainer = pycrfsuite.Trainer()
for x, y in zip(train_x, train_y):
    trainer.append(x, y)
trainer.train('space.crfsuite')

