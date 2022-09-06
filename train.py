import tensorflow as tf
from model import SR_QP2
import pathlib
import os
import glob
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import time
import utilty as util
import cv2


BATCH_SIZE = 48
BUFFER_SIZE=1000
os.environ["CUDA_VISIBLE_DEVICES"]="0"


data_dir = './batch_data/BVI_42/'
data_dir = pathlib.Path(data_dir)

in_dir=os.path.join(data_dir, 'input')
#in_dir=pathlib.Path(in_dir)

bic_dir=os.path.join(data_dir, 'interpolated')
#bic_dir=pathlib.Path(bic_dir)

true_dir=os.path.join(data_dir, 'true')
#true_dir=pathlib.Path(true_dir)





image_in=sorted(glob.glob(in_dir+"/*.png"))
image_true=sorted(glob.glob(true_dir+"/*.png"))
image_bic=sorted(glob.glob(bic_dir+"/*.png"))

image_count = len(image_bic)



def map_func(img_in, img_true, img_bic):
    img_tensor_in=tf.io.read_file(img_in)
    img_tensor_in=tf.image.decode_png(img_tensor_in,channels=1,dtype=tf.dtypes.uint16)
    img_tensor_in=tf.image.convert_image_dtype(img_tensor_in, tf.float32)

    img_tensor_true = tf.io.read_file(img_true)
    img_tensor_true = tf.image.decode_png(img_tensor_true, channels=1,dtype=tf.dtypes.uint16)
    img_tensor_true = tf.image.convert_image_dtype(img_tensor_true, tf.float32)

    img_tensor_bic = tf.io.read_file(img_bic)
    img_tensor_bic = tf.image.decode_png(img_tensor_bic, channels=1,dtype=tf.dtypes.uint16)
    img_tensor_bic = tf.image.convert_image_dtype(img_tensor_bic, tf.float32)

    return img_tensor_in, img_tensor_true,img_tensor_bic


options = tf.data.Options()
options.experimental_optimization.map_fusion = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.noop_elimination = True
options.experimental_optimization.apply_default_optimizations = True

dataset = tf.data.Dataset.from_tensor_slices((image_in, image_true,image_bic))
dataset=dataset.shuffle(BUFFER_SIZE)
val_size = 0
train_size=image_count-val_size

train_ds=dataset


train_ds = train_ds.map(lambda item1, item2, item3: tf.numpy_function(
          map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Shuffle and batch




train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.with_options(options)


layer= np.array([48, 48, 24, 24, 24, 12, 12, 8],dtype=np.int)

super_reslution=SR_QP2(kernel_size=3,layers_set=layer,transpose_conv=False,BatchNorm=False)



@tf.function
def loss_mse(y_true, y_pred):
    diff = tf.subtract(y_true, y_pred)
    mse = tf.reduce_mean(tf.square(diff))
    return mse





for (batch, (img_in, img_true,img_bic)) in enumerate(train_ds):
    diff = tf.subtract(img_true, img_bic, "diff")
    mse = loss_mse(img_true,img_bic)
    print(mse.numpy())
    break





lr=1e-04

optimizer = tf.keras.optimizers.Adam(learning_rate=lr,epsilon=1e-08,clipnorm=5) #clipnorm=1,clipvalue=5000
start_epoch = 0
EPOCHS = 250

IMG_SHAPE = (None, None, 1)

inputs = tf.keras.Input(shape=IMG_SHAPE)
predictions=super_reslution(inputs)

full_model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

full_model.summary()

@tf.function
def train_step(input_img, target_img,bic_img):

    with tf.GradientTape() as tape:
        out_sr=full_model(input_img)

        mse=loss_mse(target_img,tf.add(out_sr,bic_img))
        loss_img=mse


    trainable_variables = full_model.trainable_variables
    gradients = tape.gradient(loss_img, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return mse,loss_img

folder_checkpoint='checkpoints/BIV_QP42_L2_96_48'
os.makedirs(folder_checkpoint,exist_ok=True)
checkpoint_path = folder_checkpoint+"/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
'''
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest="checkpoints/20210320/cp-0060.ckpt"
full_model.load_weights(latest)
'''

# for i, image_path in enumerate(sorted(glob.glob("Input/QP37_test/scaled/*.png"))):
#     image = util.load_image(image_path, print_console=False)
#     out = util.apply_SR(model=full_model, input_image=image, scale=2, bit=16)
#
#     name = os.path.basename(image_path)
#     file_root = util.set_image_alignment(util.load_image('Input/QP37_test/or/' + name, print_console=False),
#                                          2)  # util.load_image('Input/Test_data/or/' + name, print_console=False)
#     file_root_y = util.convert_rgb_to_y(file_root, bit=16)
#     out_y = util.convert_rgb_to_y(out, bit=16)
#
#     ps, _ = util.compute_psnr_and_ssim(out_y, file_root_y)
#     if i == 1: ps2 = ps
#     print("image {}: {}".format(i + 1, ps))
#



for (batch, (img_in, img_true,img_bic)) in enumerate(train_ds):
    print(img_true.shape)
    break


for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_img_loss = 0
    total_mse_loss = 0
    PSNR = []
    if ((epoch+1)%5==0 and epoch>0):
        lr=lr/2
        optimizer.learning_rate.assign(lr)

    for (batch, (img_in, img_true,img_bic)) in enumerate(train_ds):
        mse_loss, img_loss = train_step(img_in, img_true,img_bic)
        total_img_loss += img_loss
        total_mse_loss +=mse_loss


        if batch % 5000 == 0 and batch>0:
            print('Epoch {} Batch {} MSE {} VGG {}'.format(
                epoch + 1, batch,total_mse_loss/(batch+1), total_img_loss/(batch+1)))


            print(optimizer._decayed_lr('float32').numpy())

    ps2=0
    for i, image_path in enumerate(sorted(glob.glob("Input/QP37_test/scaled/*.png"))):
        image = util.load_image(image_path, print_console=False)
        out = util.apply_SR_backup(model=full_model, input_image=image, scale=2,bit=16)

        name = os.path.basename(image_path)
        file_root = util.set_image_alignment(util.load_image('Input/QP37_test/or/' + name, print_console=False), 2)#util.load_image('Input/Test_data/or/' + name, print_console=False)
        file_root_y=util.convert_rgb_to_y(file_root,bit=16)
        out_y=util.convert_rgb_to_y(out,bit=16)

        ps, _ = util.compute_psnr_and_ssim(out_y, file_root_y)
        if i==1: ps2=ps
        print("image {}: {}".format(i + 1, ps))
        PSNR.append(ps)


    print('Epoch {} VALID PSNR {}'.format(
        epoch + 1, np.mean(PSNR)))


    if epoch % 1 == 0:
        full_model.save_weights(checkpoint_path.format(epoch=(epoch+1)))
    #print(full_model.get_layer('conv1').weights)
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    if lr<1e-8: break








