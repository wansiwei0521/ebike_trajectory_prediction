# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-04-07 21:53:16
LastEditTime: 2022-04-08 15:52:23
LastEditors: Vansw
Description: 
FilePath: //ebike_trajectory_prediction//units//destination_estimate.py
"""

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import time
import os
from tqdm import tqdm


def acc(_logits, y_batch):
    # return np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(_logits, 1), tf.convert_to_tensor(y_batch, tf.int64)), tf.float32), name='accuracy'
    )

def destination_train(trajs,destination_func,save_path):
    x_data,y_data = destination_genarate_datasets(trajs)
    
    # print(x_data)
    # print(len(x_data),len(y_data))
    x_data = np.array(x_data,dtype=np.float32)
    y_data = np.array(y_data,dtype=np.float32)
    
    np.random.seed(116)
    np.random.shuffle(x_data)
    np.random.seed(116)
    np.random.shuffle(y_data)
    tf.random.set_seed(116)
    
    threshould = int(len(x_data)*0.7)
    x_train = x_data[:threshould]
    y_train = y_data[:threshould]
    x_val = x_data[-threshould:]
    y_val = y_data[-threshould:]
    
    ## start training
    n_epoch = 200
    batch_size = 64
    print_freq = 5
    train_weights = destination_func.trainable_weights
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)

    ## the following code can help you understand SGD deeply
    for epoch in tqdm(range(n_epoch)):  ## iterate the dataset n_epoch times
        start_time = time.time()
        ## iterate over the entire training set once (shuffle the data via training)
        for X_batch, y_batch in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=True):
            destination_func.train()  # enable dropout
            with tf.GradientTape() as tape:
                ## compute outputs
                _logits = destination_func(X_batch)
                ## compute loss and update model
                _loss = tl.cost.normalized_mean_square_error(_logits, y_batch, name='train_loss')
                
            grad = tape.gradient(_loss, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print(f"train loss: {_loss}")
        ## use training and evaluation sets to evaluate the model every print_freq epoch
        # if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        #     destination_func.eval()  # disable dropout
        #     print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        #     train_loss, train_acc, n_iter = 0, 0, 0
        #     for X_batch, y_batch in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=False):
        #         _logits = destination_func(X_batch)
        #         train_loss += tl.cost.normalized_mean_square_error(_logits, y_batch, name='eval_loss')
        #         train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
        #         n_iter += 1
        #     print("   train foo=1 loss: {}".format(train_loss / n_iter))
        #     print("   train foo=1 acc:  {}".format(train_acc / n_iter))
            

    if not os.path.isdir(save_path):
                os.makedirs(save_path)
    save_path = save_path + "/destination.npz"
    tl.files.save_npz(destination_func.trainable_weights, name=save_path)
    
def destination_genarate_datasets(trajs):
    

    x = np.full([1,4], np.nan)
    y = np.full([1,2], np.nan)
    # print(x,y)

    for traj in trajs:
        
        x_ = traj[:,0:4]
        y_ = traj[-1][0:2] + traj[:,0:2]
        # print(y_)
        
        x = np.append(x,x_,axis=0)
        y = np.append(y,y_,axis=0)

    
    x = x[1:]
    y = y[1:]
    # print(x,y)
    # x = x[~np.isnan(x)]
    # y = y[~np.isnan(y)]
    return x,y

