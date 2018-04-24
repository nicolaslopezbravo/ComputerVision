import time
# Nicolas Lopez Bravo PA4 Robot Vision
import tensorflow as tf

########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        
        X_ = tf.contrib.layers.flatten(X)
        dense = tf.layers.dense(X_, hidden_size, activation=tf.nn.sigmoid)
        # we need 10 units for 0-9
        fcl = tf.layers.dense(dense, 10)
    
        return fcl

    # Use two convolutional layers, using strides = 1, kernel = 5, pool = 2
    def model_2(self, X, hidden_size):
        
        firstConvLayer = tf.layers.conv2d(X,filters=20,kernel_size=5,activation=
        tf.nn.sigmoid)
        firstConvLayer = tf.layers.max_pooling2d(firstConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.layers.conv2d(firstConvLayer,filters=40,kernel_size=5,activation=
        tf.nn.sigmoid)
        secondConvLayer = tf.layers.max_pooling2d(secondConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.contrib.layers.flatten(secondConvLayer)
        
        dense = tf.layers.dense(secondConvLayer, hidden_size, activation=tf.nn.sigmoid)
        # we need 10 units for 0-9
        
        fcl = tf.layers.dense(dense, 10)
       
        return fcl

    
    def model_3(self, X, hidden_size):
        # Just Replace sigmoid with ReLU.
        firstConvLayer = tf.layers.conv2d(X,filters=20,kernel_size=5,padding='SAME',activation=
        tf.nn.relu)
        firstConvLayer = tf.layers.max_pooling2d(firstConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.layers.conv2d(firstConvLayer,filters=40,kernel_size=5,padding='SAME',activation=
        tf.nn.relu)
        secondConvLayer = tf.layers.max_pooling2d(secondConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.contrib.layers.flatten(secondConvLayer)
        
        dense = tf.layers.dense(secondConvLayer, hidden_size, activation=tf.nn.relu)
        
        # we need 10 units for 0-9
        fcl = tf.layers.dense(dense, 10)
               
        return fcl

    
    def model_4(self, X, hidden_size, decay):
        # Just Replace sigmoid with ReLU.
        firstConvLayer = tf.layers.conv2d(X,filters=20,kernel_size=5,padding='SAME',activation=
        tf.nn.relu)
        firstConvLayer = tf.layers.max_pooling2d(firstConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.layers.conv2d(firstConvLayer,filters=40,kernel_size=5,padding='SAME',activation=
        tf.nn.relu)
        secondConvLayer = tf.layers.max_pooling2d(secondConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.contrib.layers.flatten(secondConvLayer)
        
        fcl1 = tf.layers.dense(secondConvLayer, hidden_size, activation=tf.nn.relu)
        
        # L2 regularizer
        l2 = tf.contrib.layers.l2_regularizer(decay)
        
        # Add one extra fully connected layer.
        fcl2 = tf.layers.dense(fcl1,hidden_size,activation = tf.nn.relu,kernel_regularizer=l2)
        
        # we need 10 units for 0-9
        fcl2 = tf.layers.dense(fcl2, 10)
        return fcl2

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # Just Replace sigmoid with ReLU.
        firstConvLayer = tf.layers.conv2d(X,filters=20,kernel_size=5,padding='SAME',activation=
        tf.nn.relu)
        firstConvLayer = tf.layers.max_pooling2d(firstConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.layers.conv2d(firstConvLayer,filters=40,kernel_size=5,padding='SAME',activation=
        tf.nn.relu)
        secondConvLayer = tf.layers.max_pooling2d(secondConvLayer,pool_size=2,strides=1)
        
        secondConvLayer = tf.contrib.layers.flatten(secondConvLayer)
        
        fcl1 = tf.layers.dense(secondConvLayer, hidden_size, activation=tf.nn.relu)
        
        # Add one extra fully connected layer.
        fcl2 = tf.layers.dense(fcl1,hidden_size,activation = tf.nn.relu)
        
        # set drop rate of .5
        dropoutRegularization = tf.layers.dropout(fcl2,.5,training=is_train)
        
        # we need 10 units for 0-9
        fcls = tf.layers.dense(dropoutRegularization, 10)
        return fcls

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)

        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.int32, [None])
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)

            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)


            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

           
            logits = features
            
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits))

            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            correct = tf.nn.in_top_k(logits, Y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = True
            if has_GPU:
                gpu_option = tf.GPUOptions(allow_growth = True)
                #per_process_gpu_memory_fraction=0.3
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))
                    test_accuracy = accuracy.eval(feed_dict={X: testX, Y: testY, is_train: False})
                    #total_correct = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    #print ('accuracy of the trained model %f' % (total_correct / testX.shape[0]))
                    print ('accuracy of the trained model %f' % (test_accuracy))
                    print ()

                return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False}) / testX.shape[0]
                #return accuracy.eval(feed_dict={X: testX, Y: testY, is_train: False})



