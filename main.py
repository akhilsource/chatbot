from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from keras.layers import *
import pickle
import numpy as np

tokenizer = Tokenizer(filters=[])
with open("D:/chat_bot/train_qa.txt", 'rb') as f:
    td = pickle.load(f)
with open("D:/chat_bot/test_qa.txt", 'rb') as fp:
    te_d = pickle.load(fp)

vocab = set()
all_data = td + te_d
for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
vocab.add('yes')
vocab.add('no')
vocab_len = len(vocab) + 1
max_story_len = max([len(data[0]) for data in all_data])
max_ques_len = max([len(data[1]) for data in all_data])
print(vocab_len, max_story_len, max_ques_len)
tokenizer.fit_on_texts(vocab)
print(tokenizer.word_index)
train_story = []
train_question = []
train_answers = []
for story, question, answer in td:
    train_story.append(story)
    train_question.append(question)
train_story_seq = tokenizer.texts_to_sequences(train_story)
train_question_seq = tokenizer.texts_to_sequences(train_question)
def vector(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_ques_len=max_ques_len):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1
           
        X.append(x)
        Xq.append(xq)
        Y.append(y)
   

    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_ques_len), np.array(Y))


input_train, question_train, answer_train = vector(td)
input_test, question_test, answer_test = vector(te_d)
print(input_train)
print(question_test)
input_sequence = Input((max_story_len),)
question = Input((max_ques_len),)
input_encod_m = Sequential()
input_encod_m.add(Embedding(input_dim=vocab_len, output_dim=64))
input_encod_m.add(Dropout(0.3))

input_encod_c = Sequential()
input_encod_c.add(Embedding(input_dim=vocab_len, output_dim=max_ques_len))
input_encod_c.add(Dropout(0.3))

question_encod = Sequential()
question_encod.add(Embedding(input_dim=vocab_len, output_dim=64, input_length=max_ques_len))
question_encod.add(Dropout(0.3))

question_encoded = question_encod(question)
input_encoded_c = input_encod_c(input_sequence)
input_encoded_m = input_encod_m(input_sequence)

match = dot([input_encoded_m, question_encoded],axes=(2,2))
match = Activation('softmax')(match)
#input_encoded_c = Reshape((max_story_len, max_ques_len))(input_encoded_c)
response=add([match,input_encoded_c])
response=Permute((2,1))(response)
answer=concatenate([response,question_encoded])
print(answer)
answer=LSTM(32)(answer)
answer=Dropout(0.5)(answer)
answer=Dense(vocab_len)(answer)
answer=Activation('softmax')(answer)
model=Model([input])
model=Model([input_sequence,question],answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
history=model.fit([input_train,question_train],answer_train,batch_size=32,epochs=100,validation_data=([input_test,question_test],answer_test))
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accrracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()
model.save('chatbot_model')
model.load_weights("chatbot_model")
predict=model.predict(([input_test,question_test]))
#' '.join([te_d[20][0]])
#' '.join([te_d[20][1]])
k=0
val_max=np.argmax(predict[20])
for key,val in tokenizer.word_index.items():
    if val==val_max:
        k=key
print("predicyed answer is",k)
print("proabity of certainity ",predict[20][val_max])


from sklearn.metrics import accuracy_score
pred=[]
for i in range(len(predict[:])):
    val_max=np.argmax(predict[[i]])
    for key,val in tokenizer.word_index.items():
        if val==val_max:
            k=key
            pred.append(key)
ans_set=[]
for story,que,ans in te_d:
    ans_set.append(ans)
a=accuracy_score(ans_set,pred)
print(a) 