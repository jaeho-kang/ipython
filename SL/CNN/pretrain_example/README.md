#Eager Execute, Pretrain, Transfer Learning

Mnist 분류기 모델을 만들고 Fully Connect 레이어를 떼고 모델을 저장하고,

저장된 파일을 로딩해서 Fully Connect 레이어를 추가시키고 재학습. 



- 분류기 모델을 만들고, Fully Connect 레이어를 떼고 모델을 저장하는 예제 
  - [mnist-eager-class-save_model_N_weights.ipynb](https://github.com/jaeho-kang/ipython/blob/master/SL/CNN/pretrain_example/mnist-eager-class-save_model_N_weights.ipynb)

- 저장된 모델을 로딩하고, Fully Connect 레이어를 추가하고, Transfer Learning 수행. 
  - [mnist_pretrain.ipynb](https://github.com/jaeho-kang/ipython/blob/master/SL/CNN/pretrain_example/mnist_pretrain.ipynb)



## Code 일부 

- 학습된 모델 저장.

  `import tensorflow.contrib.eager as tfe`

  `tfe.Saver((model.variables)).save("./model/mnist.cpk")`

- 학습된 모델에서 마지막 노드 제거 

  `model.layers.pop()`

- 저장된 모델을 로딩 

  `import tensorflow.contrib.eager as tfe`

  `tfe.Saver((model.variables)).restore("./model/mnist.cpk")`

  위 저장시키는 모델의 레이어와 로딩되는 모델의 레이어가 맞지 않으면 오류 발생. 

- Pretrain model 로딩해서 레이어 추가

- ```python
  class Mnist_with_out_top(tf.keras.Model):
      def __init__(self):
          super(Mnist_with_out_top, self).__init__()
          self.conv1 = layers.Conv2D(16,[3,3], activation='relu')
          self.conv2 = layers.Conv2D(16,[3,3], activation='relu')
          self.conv3 = layers.Conv2D(16,[3,3], activation='relu')
          self.flat = layers.Flatten()
          
      
      def __call__(self, x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.conv3(x)
          x = self.flat(x)
          return x
      
  #define class mnist classification with pretain
  class Mnist_with_Pretrain(tf.keras.Model):
      def __init__(self, pretrain):
          super(Mnist_with_Pretrain, self).__init__()
          self.pretrain = pretrain # pretrain 모델을 객체로 받아옴.
          self.pretrain.trainable = False # 해당 모델은 재학습을 수행하지 않음.
          self.dense = layers.Dense(10)
      
      def __call__(self, x):
          x = self.pretrain(x) # pretain 모델을 이용.
          x = self.dense(x)
          return x
      
  model = Mnist_with_out_top() # Fully Connect Layer가 없는 객체 선언.
  tfe.Saver((model.variables)).restore("./model/mnist.cpk")# 저장된 모델 정보를 로딩. 
  new_model = Mnist_with_Pretrain(model) # Fully Connect Layer가 있는 객체 생성
  
  x = tf.random_normal([1,28,28,1])
  print(new_model(x).shape) # Fully Connect를 통해 완성된 모델 체크
  ```

- 학습 코드 

  ```python
  device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'
  print('using device %s' %device)
  epoch=2
  loss_history = []
  acc_history = []
  with tf.device(device):  
  #--- 중략 ---
  	#tape.gradient에서 두번째 인자를 trainable_variables를 사용해야함!!
      grads = tape.gradient(loss_value, new_model.trainable_variables)
      #tape.optimizer.apply_gradients 두번째 인자를 trainable_variables를 사용해야함!!
      optimizer.apply_gradients(zip(grads, new_model.trainable_variables), 
                                global_step=tf.train.get_or_create_global_step())
  
      #위에서 new_model.trainable_variables를 사용하지 않고, new_model.variables을 사용할 경우, 
      # new_model에서 pretrain객체의 레이어 들도 학습이 수행됨.
  ```


## eager execute 모델에서 pretrain을 이용한 transer learning이 이 방식이 정석인지 잘 모르곘습니다. 

하다보니깐 되서 우선 정리 했는데. 오류나 뭔가 맞지 않다면 이슈로 남겨 주시면 반영하도록 하겠습니다. 

