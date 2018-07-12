import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt


from PIL import Image

#getting the data
num_training=2400
num_validation=100

X_train = np.zeros((2500,128,128))
y_train = np.zeros((2500,128,128))

for i in range(1,51):
    for j in range(1,51):
        #print(i,j)
        imnoisy = np.array(Image.open('noisy_lines/nline_'\
                                      +str(i) +'_'+str(j)+'.tiff'))
        im = np.array(Image.open('original_lines/line'\
                                 +str(i)+'.tiff'))
        index = (i-1)*50+j-1
        #print(index)
        X_train[index] = imnoisy
        y_train[index] = im



X_train = np.reshape(X_train,(2500,128,128,1))
y_train = np.reshape(y_train,(2500,128,128,1))

X_train = X_train/256
y_train = y_train/256

mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(y_train))

print (X_train.dtype)
print (y_train.dtype)


print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=10,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    #correct_prediction = tf.equal(tf.argmax(predict,1), y)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = 0
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    # variables = [mean_loss,correct_prediction,accuracy]
    variables = [mean_loss,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        #correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx,:],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            #correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.6g}"\
                      .format(iter_cnt,loss))
            iter_cnt += 1
        #total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {1}, Overall loss = {0:.3g}"\
              .format(total_loss,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            #plt.show()
    return total_loss
	
def my_model(X,y,is_training):
    
    conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch1 = tf.layers.batch_normalization(conv1, axis=3, training=is_training)
    
    conv2 = tf.layers.conv2d(inputs= batch1,filters= 32,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch2 = tf.layers.batch_normalization(conv2, axis=3, training=is_training)

    conv3 = tf.layers.conv2d(inputs= batch2,filters= 32,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch3 = tf.layers.batch_normalization(conv3, axis=3, training=is_training)
    
    conv4 = tf.layers.conv2d(inputs= batch3,filters= 32,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch4 = tf.layers.batch_normalization(conv4, axis=3, training=is_training)
    
    conv5 = tf.layers.conv2d(inputs= batch4,filters= 32,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch5 = tf.layers.batch_normalization(conv5, axis=3, training=is_training)
    
    conv6 = tf.layers.conv2d(inputs= batch5,filters= 32,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch6 = tf.layers.batch_normalization(conv6, axis=3, training=is_training)
    
    conv7 = tf.layers.conv2d(inputs= batch6,filters= 32,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch7 = tf.layers.batch_normalization(conv7, axis=3, training=is_training)
    
    conv8 = tf.layers.conv2d(inputs= batch7,filters= 32,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
    batch8 = tf.layers.batch_normalization(conv8, axis=3, training=is_training)
    
    #dropout2 = tf.layers.dropout(h2,rate = 0.25, training=is_training)
    y_out = tf.layers.conv2d(inputs= batch8,filters= 1,kernel_size=[3, 3],padding="same")
    
    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 128, 128, 1])
y = tf.placeholder(tf.float32, [None, 128, 128, 1])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
total_loss = tf.losses.mean_squared_error(y, y_out)

mean_loss = tf.reduce_mean(total_loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = 1e-4
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                           200, 0.96, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate)      #ORIGINAL = 5e-4

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

# Feel free to play with this cell
# This default code creates a session
# and trains your model for 10 epochs
# then prints the validation set accuracy

batch_size = 20
epochs = 1

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,epochs,batch_size,10,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,batch_size)
