import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

model = tf.global_variables_initializer();


data = read_csv('https://raw.githubusercontent.com/aimarkhwang/aiam202010/main/csv/corona.csv')
#csv 값을 읽어와서 변수 data 로 선언
xy = np.array(data, dtype=np.float32)
#numpy함수를 이용해서 numpy.array로 담아둠.

x = xy[:, 1:-1]
y = xy[:, [-1]]
#x(확진자수) , y(단계)

X = tf.placeholder(tf.float32, shape=None)
Y = tf.placeholder(tf.float32, shape=None)
# x와 y의 값을 담아줄 placeholder 설정 
W = tf.Variable(tf.random_normal([1]), 'weight')
b = tf.Variable(tf.random_normal([1]), 'bias')
#임의의 random한 값을 가진 W(weight),b(bias) 설정
hypothesis = X * W + b
#가설 값 설정 W는 기울기 b는 y절편 값
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost 함수 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(cost)
# 경사하강법 최적화 함수 학습률 0.00005 더 올리면 inf 무한대로 치솟은 overshooting이 일어납니다.(learnig rate = W를 미분할때 알파값에 해당)



sess = tf.Session()
#세션을 생성
sess.run(tf.global_variables_initializer())
#글로벌 변수를 초기화
costhistory=[]

for step in range(10001):
    
    cost_, hypo_, tr_ = sess.run([cost, hypothesis, train], feed_dict={X: x, Y: y})

    if step % 100 == 0:

        print("epoch :", step, ", cost :", cost_)
        print("예측치: ", hypo_[0])
        costhistory.append(cost_)
# feed_dict x와 y의 값을 palcehoder X,Y에 담아주는 feed_dict argument를 이용하여 학습 /cost의 시각화를 위해 costhistory.append(cost_) 담음.
saver = tf.compat.v1.train.Saver()
# saver 함수를 이용해 값을 저장
save_path = saver.save(sess, "./model/saved.cpkt")
#지정된 경로에 saver를 저장
print('learning is complete')

plt.figure(figsize=[5,5])
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('epoch/cost')
plt.plot(costhistory)

plt.show()
sess.close()