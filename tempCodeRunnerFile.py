history=model.fit([input_train,question_train],answer_train,batch_size=32,epochs=100,validation_data=([input_test,question_test],answer_test))
