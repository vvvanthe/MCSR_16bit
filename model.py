import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import Constant
class SR_QP2(tf.keras.Model):
    def __init__(self, kernel_size,layers_set,drop_rate=0.1,transpose_conv=True,BatchNorm=True,alpha=0.2):
        super().__init__()
        self.drop_rate=drop_rate
        self.kernel_size=kernel_size
        self.layers_set=layers_set
        self.transcov2=transpose_conv
        self.BatchNorm=BatchNorm
        self.apl=alpha
        self.input_feature_num=1


    def __call__(self, inputs):
        list_out = []
        x = inputs

        for i in range(len(self.layers_set)):
            if i==0:
                self.kernel_size=9
            elif i==1 or i==2:
                self.kernel_size=6
            else:
                self.kernel_size=3

            base_name='CNN' + str(i+1)
            x = tf.keras.layers.Conv2D(filters=self.layers_set[i], use_bias=True, kernel_size=self.kernel_size, padding='same',
                                       name=base_name, bias_initializer='zeros')(x)

            x = tf.keras.layers.LeakyReLU(alpha=self.apl)(x)


            if self.drop_rate>0:
                x=tf.keras.layers.Dropout(self.drop_rate)(x)

            if self.BatchNorm:
                x=tf.keras.layers.BatchNormalization(trainable=False)(x)

            list_out.append(x)

            self.input_feature_num = self.layers_set[i]

        H_concat = tf.concat(list_out, -1, name="H_concat_1")

        x=self.build_pixel_shuffler_layer(name='shuffer_1',input=H_concat)


        if not self.transcov2:
            return x

        # Transpose layer
        trans_layers = tf.keras.layers.Conv2DTranspose(filters=1, use_bias=True, kernel_size=self.kernel_size,
                                                    padding='same', name='transpose2D', strides=(2, 2))(H_concat)

        trans_layers = tf.keras.layers.LeakyReLU(alpha=self.apl)(trans_layers)
        #if self.drop_rate > 0:
        #    trans_layers = tf.keras.layers.Dropout(self.drop_rate)(trans_layers)

        if self.BatchNorm:
            trans_layers = tf.keras.layers.BatchNormalization(trainable=False)(trans_layers)


        #Concantenate Shuffer and Transpose

        H_concat_2 = tf.concat([x,trans_layers], -1, name="H_concat_2")

        # Combine Shuffer and Transpose Layers with 1x1 Conv

        out_put = tf.keras.layers.Conv2D(filters=1, use_bias=True, kernel_size=1,
                                   padding='same',
                                   name='CNN_1x1', bias_initializer='zeros')(H_concat_2)


        if self.BatchNorm:
            out_put = tf.keras.layers.BatchNormalization(trainable=False)(out_put)


        return out_put




    def he_initializer(self,filter_num):
        n = self.kernel_size * self.kernel_size * filter_num
        stddev = np.sqrt(2.0 / n)
        return stddev


    def build_pixel_shuffler_layer(self,name,input,activator=False):
        init_value =tf.keras.initializers.Constant(self.he_initializer(self.input_feature_num))
        output = tf.keras.layers.Conv2D(filters=4, use_bias=True, kernel_size=self.kernel_size,
                                   padding='same',
                                   name=name+'_CNN', bias_initializer='zeros')(input)


        output=tf.nn.depth_to_space(output, 2)
        #output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
        if activator:
            output = tf.keras.layers.LeakyReLU(alpha=self.apl)(output)

        return output


class prelu_v2(tf.keras.layers.Layer):
    def __init__(self,output_dim,name_lay):
        super(prelu_v2,self).__init__()
        b_init = tf.zeros_initializer()
        b_start=tf.constant(0.1,shape=(output_dim,))
        self.b = tf.Variable(
            initial_value=b_start,name=name_lay, trainable=True #b_init(shape=(output_dim,)
        )

    def call(self, inputs):

        return tf.keras.layers.ReLU()(inputs)+ tf.multiply(tf.abs(self.b), (inputs - tf.abs(inputs))) * 0.5#tf.maximum(0.0, inputs)+ tf.multiply(self.b, (inputs - tf.abs(inputs))) * 0.5

