
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle

OPTIMIZER = {'adam':tf.train.AdamOptimizer,
             'gradient':tf.train.GradientDescentOptimizer}

FILE_FORMAT = ['jpg', 'jpeg', 'png', 'bmp']

def pickle_save(d, file_name):
    with open("{}.pickle".format(file_name), 'wb') as f:
        pickle.dump(d, f)

def pickle_load(file_name):
    with open("{}.pickle".format(file_name), 'rb') as f:
        return pickle.load(f, encoding='bytes')

def image_load(file_path):
    return np.array(Image.open(file_path)).astype(float)/255.

def pprint(*message): # TODO
    print(*message)

def aassert(statement, message=''):
    if not statement:
        pprint(message)
        assert False, message

def report_plot(data, i, model_name, log='./log'):
    print(data,i)
    if not os.path.exists(log):
        os.mkdir(log)
    if i==0 or not os.path.exists(os.path.join(log,"{}.pickle".format(model_name))):
        with open(os.path.join(log,"{}.pickle".format(model_name)), 'w'):
            pass
        pickle_save([[data], [i]], os.path.join(log,model_name))
        return
    d, t = pickle_load(os.path.join(log,model_name))
    d.append(data)
    t.append(i)
    pickle_save([d,t], os.path.join(log,model_name))
    plt.plot(t,d)
    plt.pause(0.0001)

def shuffle(x, y):
    from random import shuffle as sf
    d = list(zip(x,y))
    sf(d)
    x_, y_ = zip(*d)
    return np.array(x_).astype(float), np.array(y_).astype(float)

def call_mnist(one_hot=True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)
    train_data, train_label = shuffle(np.concatenate((mnist.train.images, mnist.validation.images),axis=0),
                                     np.concatenate((mnist.train.labels, mnist.validation.labels),axis=0))
    test_data, test_label = shuffle(mnist.test.images, mnist.test.labels)
    return train_data.reshape(-1, 28,28,1), train_label, test_data.reshape(-1, 28,28,1), test_label

def get_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,1), tf.argmax(y_pred, 1)), tf.float32))

def data_input(data, istrain, label=None):
    """
        Args
        - data : data(numpy array) or directory with data(str).
        - label : (optional)
        Return
        - [data] [file_path] [label(option)] [image_size] # we assume that label is not one-hot code configuration.
    """

    if type(data) == str: # directory
        file_path, label = dir2label(data, istrain)
        data = None
        image_size = list(image_load(file_path[0]).shape)
    elif type(data) == np.ndarray:
        data = data if len(data.shape) == 4 else data[:,:,:,np.newaxis]
        if type(label) == str:
            file_path, label = txt2label(label, istrain)
        else:
            file_path = None
        image_size = data.shape
    else:
        aassert (False, "The data has to be data itself(numpy array) or directory(string) containing data. But {}".format(type(data)))
    # pprint(label)
    if label is not None and np.max(label) > 0:
        label = one_hot_coding(label)
    return data, file_path, label, image_size

def txt2label(file_path, istrain):
    """
        Return
        - [file_name], [:[class]] : if data is located in own class directory.
        - None, [:[class]] : if all data is aggregated in a directory.
    """
    label, line = [], 0
    could = os.path.exists(file_path)
    if not could and not istrain:
        return None, None
    aassert(could or not istrain)

    with open(file_path, 'r') as f:
        while True:
            instance = f.readline().split('\n')[0].replace('\t', ' ').split(' ')
            instance = [int(i) if i.isdigit() else i for i in instance if i]
            if not instance: break
            label.append(instance)
            aassert (len(label[line]) == len(label[line-1]), "inconsistency detected on {}th line.".format(line))
            line += 1
    # line = len(label[0])
    classes_path = os.path.join(*file_path.split('/')[:-1], 'classes')
    if istrain:
        # classes = sorted(list(set([lb[i] for lb in label ]) for i in range(line)))
        classes = sorted(list(set([lb[1] for lb in label])))
        pickle_save(classes, classes_path)
    else:
        classes = pickle_load(classes_path)
    # str2cls = [{v:i for i, v in enumerate(cls)} for cls in classes]
    # cls2str = [{i:v for i, v in enumerate(cls)} for cls in classes]
    str2cls = {v:i for i, v in enumerate(classes)}
    cls2str = {i:v for i, v in enumerate(classes)}

    if line == 1:
        return None, np.array([str2cls[instance] for instance in label]), cls2str
    elif line >= 2:
        return [instance[0] for instance in label], np.array([str2cls[instance[1]] for instance in label])


def dir2label(dataset, istrain, file_format=FILE_FORMAT):
    """
        Return
        - [file_name], [class]
    """
    mode = 'train' if istrain else 'test'
    directory = os.path.join(os.getcwd(),'dataset', dataset, 'Data')
    classes = sorted([dir for dir in os.listdir(directory) if len(dir.split('.'))==1])

    if len(classes) == 0: # data is aggregated.
        txt_path = os.path.join(directory, '{}.txt'.format(mode))
        aassert (os.path.exists(txt_path), " [@] {}.txt doesn't exist".format(mode))
        return txt2label(txt_path, istrain)

    else:
        classes_path = os.path.join(directory, 'classes')
        if istrain:
            classes = sorted([dir for dir in os.listdir(directory) if len(dir.split('.'))==1])
            pickle_save(classes, classes_path)
        else:
            classes = pickle_load(classes_path)
        file_path = [[os.path.join(directory, cls, file_name) for file_name in os.listdir(os.path.join(directory,cls)) if file_name.split('.')[-1].lower() in file_format] for cls in classes]
        label = np.concatenate([np.array([integer] * len(file_path[integer])) for integer, cls_name in enumerate(classes)])

        return sorted(list(set().union(*file_path))), label

def label2txt(label, txt_name, dataset, file_name=None):
    """
        Return :
        - [file_name] [class]\n # for each instance if file_name exists
        - [class]\n # for each instance if file_name does not exists (It implies that data has been ordered by names. )
    """
    directory = os.path.join(os.getcwd(),'dataset', dataset, 'Data')
    txt_path = os.path.join(directory, '{}.txt'.format(txt_name))
    with open(txt_path, 'w') as f:
        if file_name:
            aassert (len(label) == len(file_name))
            for i in range(len(label)):
                f.write('{} {}\n'.format(file_name[i], label[i])) # TODO : multi-label
        else:
            for i in range(len(label)):
                f.write('{}\n'.format(label[i]))
    return txt_name

def one_hot_coding(y):
    y= np.array(y)
    N = y.shape[0]
    K = np.max(y, axis=0).astype(np.int32)+1 # multiple labels can be accepted.
    one_hot = np.zeros([N, K])
    one_hot[np.arange(N), y] = 1
    return one_hot

def cross_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def linear(x, out_dim, name, stddev=0.02):
    in_dim = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [in_dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [out_dim], initializer=tf.zeros_initializer)
    return tf.nn.bias_add(tf.matmul(x, w), b)

open_data = {'mnist':call_mnist}

def load_model(sess,               # tf.Session()
               dataset = None,     # dataset name, open data e.g. mnist or hand-crafted dataset
               classes = None,     #
               image_size = None,  # if it is given, image can be cropped or scaling
               train_data = None,  # folder with images
               test_data = None,   # folder with images
               train_label = None, # txt file
               test_label = None,  # (optional) txt file
               learning_rate = 1e-4,
               optimizer = 'gradient',
               beta1 = 0.5,   # (optional) for some optimizer
               beta2 = 0.99,  # (optional) for some optimizer
               batch_size = 64,
               epochs = 20,
               checkpoint_dir = "checkpoint",
               checkpoint_name = None,
               train = False,
               test = False,
               epoch_interval = None,
               step_interval = None
               ):

    if not train and not test: return
    net = LOGISTIC(sess, dataset, classes, image_size, learning_rate, optimizer, beta1, beta2, batch_size, epochs)
    load, epoch = net.load(checkpoint_dir, checkpoint_name)
    if train: net.train(epoch, checkpoint_dir, checkpoint_name, epoch_interval, step_interval, train_data, train_label)
    if test:
        aassert(train or load, " [@] Train model first.")
        net.test(test_data, test_label)



class LOGISTIC(object):
    """

    Scenario 1 : Using open data.
     In that case, following arguments are useless
        : classes, image_size, train_data, test_data, train_label, test_label

    Scenario 2 : Hand-crafted dataset.
     In that case, following arguments are useless.
        : dataset

    """
    def __init__(self,
                 sess,               # tf.Session()
                 dataset,     # dataset name, open data e.g. mnist or hand-crafted dataset
                 classes,     #
                 image_size,  # if it is given, image can be cropped or scaling
                 learning_rate = 1e-4,
                 optimizer = 'gradient',
                 beta1 = 0.5,   # (optional) for some optimizer
                 beta2 = 0.99,  # (optional) for some optimizer
                 batch_size = 64,
                 epochs = 20
                 ):

        self.sess = sess
        self.classes = classes
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.build_model()



    def get_batch(self, data, label, i, data_file=None, batch_size = None):
        if not batch_size : batch_size = self.batch_size
        last_batch = False if i < self.no_batch else True
        batch_offset = self.batch_size * i
        batch_size = self.batch_size if not last_batch else len(label)-batch_offset
        if data is not None:
            batch_x = data[batch_offset:batch_offset+batch_size]
        else:
            batch_file = data_file[batch_offset:batch_offset+batch_size]
            batch_x = np.array([image_load(data_file) for data_file in batch_file]) # TODO : crop

        if label is not None:
            batch_y = label[self.batch_size*i:self.batch_size*(i+1)] if not last_batch else label[self.batch_size*i:]
        else:
            batch_y = None

        return batch_x, batch_y

    def build_model(self):

        self.X = tf.placeholder(tf.float32, [None]+self.image_size, name="input")
        self.Y = tf.placeholder(tf.float32, [None]+[self.classes], name='label')

        self.y_logit, self.y_pred = self.classifier()

        self.loss = cross_entropy(self.Y, self.y_logit)
        self.acc = get_accuracy(self.Y, self.y_logit)
        self.prediction = tf.argmax(self.y_pred, axis=1)

        if self.optimizer == 'gradient':
            self.optim = OPTIMIZER[self.optimizer](self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'adam':
            self.optim = OPTIMIZER[self.optimizer](self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.loss)
        else:
            raise NotImplementedError('.')

        self.saver = tf.train.Saver()


    """
    #1 By dictionary
        Train Data path : [Project Folder]/dataset/[data name]/Data/[Class name]/[Data_name].[format]
        Test Data path : [Project Folder]/dataset/[data name]/Data/test/[Data_name].[format]
    #2 By txt
        Data path : [Project Folder]/dataset/[data name]/Data/[Data_name].[format]
    Checkpoint directory : [Project Folder]/checkpoint/[Model_name]/[Epoch or Iteration].ckpt

    """
    def save(self, checkpoint_dir, epoch, name=None):

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        file_name = 'model.ckpt'

        self.saver.save(self.sess, os.path.join(checkpoint_dir,file_name), global_step=epoch)

    def load(self, checkpoint_dir, name=None):
        import re
        pprint(" [*] Reading checkpoints....")
        self.model_name = name if name else self._model_name
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        if os.path.exists(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                pprint(" [!] checkpoint name is {}".format(ckpt_name))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0)) if not name else 0 # TODO
                pprint(" [*] Success to read {}".format(ckpt_name))
                return True, counter+1
            else:
                pprint(" [@] Failed to read the checkpoint.")
                aassert(False)
        else:
            pprint(" [@] Failed to find the checkpoint.")
            self.sess.run(tf.global_variables_initializer())
            return False, 0

    def classifier(self, reuse=False):
        im_dim = np.product(np.array(self.image_size))
        with tf.variable_scope('classifier') as scope:
            if reuse:
                scope.reuse_variables()
            X = self.X if len(self.X.get_shape().as_list())==2 else tf.reshape(self.X, [-1, im_dim])
            h = linear(X, self.classes, 'linear') # TODO : multi-classes
        return h, tf.nn.softmax(h)

    def train(self, epoch_trained, checkpoint_dir, checkpoint_name, epoch_interval=None, step_interval=None,
              train_data = None, train_label = None):

        plt.ion()
        plt.title('loss')
        fig=plt.figure('{}_training '.format(self.model_name))
        pprint(" [*] Train data preprocessing...")
        if self.dataset in open_data.keys():
            self.train_data, self.train_label, self.test_data, self.test_label = open_data[self.dataset]()
            real_shape = list(self.train_data.shape[1:])
            aassert(self.image_size == real_shape, "unmatch {} vs {}".format(self.image_size, real_shape)) # TODO : cropping
            if len(self.image_size) == 2:
                self.image_size += [1]
                self.train_data = self.train_data[:,:,:,np.newaxis]
                self.test_data = self.test_data[:,:,:,np.newaxis]
            aassert(self.classes == self.train_label.shape[-1], " [!] Invaild classes")
            self.N = self.train_label.shape[0]
            self.train_data_path, self.test_data_path = None, None
        else:
            # pprint(type(train_data), type(train_label))
            self.train_data, self.train_data_path, self.train_label, real_shape = data_input(train_data,True, train_label)
            # pprint(self.train_label)
            aassert(self.image_size == real_shape, "unmatch {} vs {}".format(self.image_size, real_shape)) # TODO : cropping
            aassert(self.classes == int(self.train_label.shape[-1]),
                     " [!] Invalid classes {} vs {}, {}".format(type(self.classes), type(self.train_label), self.train_label.shape[-1]))
            self.N = self.train_data.shape[0] if self.train_data else len(self.train_data_path)
        pprint(" [!] Train data preprocessing... is done.")

        self.no_batch = int(np.ceil(self.N/self.batch_size))

        global_step = 0
        # pprint(self.epochs, epoch_trained)
        if self.epochs <= epoch_trained:
            pprint(" [!] Training is already done.")
            return
        pprint(" [*] Training start...")
        for epoch in range(self.epochs-epoch_trained):
            for step in range(self.no_batch):
                batch_x, batch_y = self.get_batch(self.train_data, self.train_label, step, self.train_data_path)
                global_step += 1
                feed_dict= {self.X:batch_x, self.Y:batch_y}
                self.sess.run(self.optim, feed_dict=feed_dict)
                if step_interval and np.mod(global_step, step_interval) == 0:
                    loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    pprint("[{}] iter, loss : {}".format(global_step, loss))
                    report_plot(loss, global_step-step_interval, self.model_name)
            if epoch_interval and np.mod(epoch, epoch_interval)==0:
                loss = self.sess.run(self.loss, feed_dict=feed_dict)
                report_plot(loss, epoch, self.model_name)
                pprint("[{}] epoch, loss : {}".format(epoch, loss))
            # self.save(checkpoint_dir, epoch, checkpoint_name)
        pprint(" [!] Trainining is done.")


    def test(self, test_data = None, test_label=None):
        """
            Result
            - file :
             [sample name or order] [True class(if labels are given)] [prediction] [is it correct?(if given)]
             ...
             [sample name or order] [True class(if labels are given)] [prediction] [is it correct?(if given)]
             The number of samples : [N], Accuracy : [acc(if labels are given)] # at last
        """
        if self.dataset not in open_data.keys():
            self.test_data, self.test_data_path, self.test_label, image_size = data_input(test_data,False, test_label)
        N = self.test_data.shape[0] if self.test_data is not None else len(self.test_data_path)
        # pprint(len(self.test_label), N)

        pprint(" [*] Test start...")
        with open(self.model_name, 'w') as f:
            correct = 0.
            for ith in range(N):
                w = [ith]
                if self.test_data is not None:
                    pred = self.sess.run(self.prediction, feed_dict={self.X:self.test_data[ith][np.newaxis]}) # TODO :multi-label
                elif self.test_data_path is not None:
                    image = image_load(self.test_data_path[ith])
                    pred = self.sess.run(self.prediction, feed_dict={self.X:image[np.newaxis]})
                if self.test_label is not None:
                    # pprint(ith)
                    w.append(int(np.argmax(self.test_label[ith])))
                w.append(pred) # TODO : multi-label
                if self.test_label is not None:
                    acc = bool(pred == int(np.argmax(self.test_label[ith])))
                    w.append(acc)
                    correct += float(acc)
                for elem in w:
                    f.write('{} '.format(elem))
                f.write('\n')
            result = "The number of samples : [{}]".format(N)
            if self.test_label is not None:
                result += ", Accuracy : [{}]".format(correct/float(N))
            f.write(result)
        pprint(" [!] Test is done.")

    @property
    def _model_name(self):
        if not self.dataset:
            self.dataset = self.file_path[0].split('/')[-4]
        return '{}_{}_{}'.format('simple', 'logistic', self.dataset)
