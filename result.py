import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv



# 플레이스 홀더를 설정합니다.

X = tf.placeholder(tf.float32, shape=None)
Y = tf.placeholder(tf.float32, shape=None)



W = tf.Variable(tf.random_normal([1]), 'weight')
b = tf.Variable(tf.random_normal([1]), 'bias')



# 가설을 설정합니다.

hypothesis = X * W + b



# 저장된 모델을 불러오는 객체를 선언합니다.

saver = tf.compat.v1.train.Saver()

model = tf.global_variables_initializer()



# 4가지 변수를 입력 받습니다.

cf = float(input('확진자 수: '))




with tf.Session() as sess:

    sess.run(model)



    # 저장된 학습 모델을 파일로부터 불러옵니다.

    save_path = "./model/saved.cpkt"

    saver.restore(sess, save_path)



    # 사용자의 입력 값을 이용해 배열을 만듭니다.

    data = ((cf), )

    arr = np.array(data, dtype=np.float32)



    # 예측을 수행한 뒤에 그 결과를 출력합니다.

    x_data = arr[0:1]

    dict = sess.run(hypothesis, feed_dict={X: x_data})

    print(dict[0])
    if dict[0] <1.2:
        print("1단계 입니다.")
    else:
        if dict[0] <1.9:
            print("2단계 입니다")
        else:
            if dict[0] <2.4:
                print("2.5단계 입니다")
        