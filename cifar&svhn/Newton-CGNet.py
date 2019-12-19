import collections
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
slim = tf.contrib.slim
import os
import random
import argparse
import sys

# 定义一个类为Block
class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
	'A namede tuple describing a ResNet block.'

def subsample(inputs,factor,scope=None):
	if factor == 1:
		return inputs
	else:
		return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
	if stride == 1:
		return slim.conv2d(inputs,num_outputs,kernel_size,stride=1,padding='SAME',scope=scope)
	else:
		pad_total = kernel_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,1]])
		return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,padding='VALID',scope=scope)

def resnet_arg_scope(weight_decay=0.0001,batch_norm_decay=.999,batch_norm_epsilon=1e-5,batch_norm_scale=True):
	batch_norm_params = {'decay': batch_norm_decay,
	'epsilon':batch_norm_epsilon,'scale':batch_norm_scale,'updates_collections': tf.GraphKeys.UPDATE_OPS}

	with slim.arg_scope(
		[slim.conv2d],
		weights_regularizer=slim.l2_regularizer(weight_decay),
		weights_initializer=slim.variance_scaling_initializer(),
		activation_fn=tf.nn.relu,
		normalizer_fn=slim.batch_norm,
		normalizer_params=batch_norm_params):
		with slim.arg_scope([slim.batch_norm],**batch_norm_params):
			with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:
				return arg_sc

def stack_blocks_dense(net,blocks,keep_prob,outputs_collections=None):
	for block in blocks:
		with tf.variable_scope(block.scope,'block',[net]) as sc:
			for i, unit in enumerate(block.args):
				# 这边可能要根据简明ResNet做改动
				with tf.variable_scope('unit_%d' %(i+1), values=[net]):
					unit_depth,unit_stride = unit #这里有改动
					net = block.unit_fn(net,
										depth=unit_depth,
										stride=unit_stride,
										keep_prob=keep_prob)
					# print('after',i+1,'block:',net.get_shape().as_list())
			# net=slim.utils.colect_named_outputs(outputs_collections,sc.name,net)
	return net

def compute_gradient(inputs,depth,keep_prob,stride=1):
	depth_in = slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
	# preact = tf.nn.relu(inputs)
	with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
		branch1 = conv2d_same(inputs,depth,3,stride)
	branch1_final = tf.nn.dropout(branch1,keep_prob)
	branch2_drop = tf.nn.dropout(inputs,keep_prob)
	with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
		branch2_final = conv2d_same(branch2_drop,depth,3,stride)
	branch3 = conv2d_same(inputs,depth,3,stride)
	branch3_drop = tf.nn.dropout(branch3,keep_prob)
	with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
		branch3_final = conv2d_same(branch3_drop,depth,3,stride=1)
	branch_sum = slim.batch_norm(branch1_final) + slim.batch_norm(branch2_final) + slim.batch_norm(branch3_final)
	if stride == 2:
		shortcut = slim.conv2d(inputs,depth,[1,1],stride=stride,activation_fn=None)
	else:
		shortcut = inputs
	outputs = shortcut + branch_sum
	# outputs = inputs + slim.batch_norm(branch_sum)
	# outputs = tf.nn.relu(outputs)

	return outputs

@slim.add_arg_scope
def bottleneck(inputs,depth,stride,keep_prob,outputs_collections=None,scope=None):
	with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
		depth_in = slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
		# preact=slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope='preact')
		if depth == depth_in:
			shortcut = subsample(inputs,stride,'shortcut')
		else:
			shortcut = slim.conv2d(inputs,depth,[1,1],stride=stride,activation_fn=None,scope='shortcut')
		# 下面是改动部分，由原来的1+3+1改为适合cifar的3+3block
		# with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
		# 	y = conv2d_same(inputs,depth,1,stride,scope='y0')
		# y = inputs
		# y = shortcut
		y = inputs
		g = compute_gradient(y,depth,keep_prob,stride)
		if depth == depth_in:
			y = subsample(y,stride,'shortcut2')
		else:
			y = slim.conv2d(y,depth,[1,1],stride=stride,activation_fn=None,scope='shortcut2')
		
		d = -g

		if depth == 16:
			k = 12
		elif depth == 32:
			k = 12
		else:
			k = 13

		for i in range(k):
			with tf.name_scope('cgblock_'+str(i+1)):
				alpha = tf.Variable(tf.constant(1.0),name='alpha')
				y = y + alpha * d
				g = compute_gradient(y,depth,keep_prob)
				beta = tf.Variable(tf.constant(1.0),name='beta')
				d = -g + beta * d
		alpha = tf.Variable(tf.constant(1.0),name='cgblock3_alpha')
		y = y + alpha * d
		outputs = shortcut + y

		# return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)
		return outputs

def resnet_v2(inputs,blocks,keep_prob,num_classes=None,is_training=True,reuse=None,scope=None):
	with tf.variable_scope(scope,'resnet_v2',[inputs],reuse=reuse) as sc:
		# with slim.arg_scope([slim.conv2d,bottleneck,stack_blocks_dense],outputs_collections=end_points_collection):
		# with slim.arg_scope([slim.batch_norm],**batch_norm_params['is_training']=is_training):
		net = inputs
		with slim.arg_scope([slim.batch_norm],is_training=is_training):
			with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
				net = conv2d_same(net,16,3,stride=1,scope='conv1')
			net = stack_blocks_dense(net,blocks,keep_prob)
			net = slim.batch_norm(net,activation_fn=tf.nn.relu,scope='postnorm')
			net = tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)
			net = slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='logits',weights_initializer=tf.contrib.layers.xavier_initializer())
				# end_points['predictions'] = slim.softmax(net,scope='predictions')
	return net #,end_points

def resnet_v2_simple(num_classes,batch_size,
	height,width,channal,reuse=None,scope='resnet_simple',n=3):
	# 初始化超参数
	with tf.name_scope('input'):
		input_x = tf.placeholder(tf.float32,[None,height,width,channal],name='input_x')
		input_y = tf.placeholder(tf.int32,[None],name='input_y')
		is_training = tf.placeholder(tf.bool,name='is_training')
		learning_rate = tf.placeholder(tf.float32,name='learning_rate')
		keep_prob = tf.placeholder(tf.float32,name='keep_prob')

	onehot_labels = tf.one_hot(input_y,depth=num_classes,axis=1)

	epoch_step = tf.Variable(0,trainable=False,name='Epoch_Step')
	iteration_step = tf.Variable(0,trainable=False,name='Iteration_Step')
	epoch_increment = tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
	iteration_increment = tf.assign(iteration_step,tf.add(epoch_step,tf.constant(1)))

	# 其中每个残差单元包含两个卷积层，args=(block_depth,stride)，
	# 两个卷积层depth一样可以共用，stride表示第一个卷积层的stride，决定是否降采样
	blocks=[
	Block('block1',bottleneck,[(16,1)] * n),
	Block('block2',bottleneck,[(32,2)]+[(32,1)]*(n-1)),
	Block('block3',bottleneck,[(64,2)]+[(64,1)]*(n-1))]

	with slim.arg_scope(resnet_arg_scope()):
		logits = resnet_v2(input_x,blocks,keep_prob,num_classes,is_training=is_training,reuse=reuse,scope=scope) 
	logits = tf.squeeze(logits,[1,2],name='squeeze')
	classification_losses = slim.losses.softmax_cross_entropy(logits,onehot_labels)
	# print('the total loss is:',slim.losses.get_regularization_losses())
	regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
	total_loss = classification_losses + regularization_loss
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate) #可以考虑改为Adam
	# optimizer = tf.train.AdamOptimizer(learning_rate)
	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(total_loss)
	predictions = tf.argmax(logits,axis=1,name='predictions')
	correct_prediction = tf.equal(tf.cast(predictions,tf.int32),input_y)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	return learning_rate,input_x,input_y,is_training,keep_prob,epoch_step,epoch_increment,iteration_step,iteration_increment,total_loss,train_op,accuracy

def do_eval(sess,evalX,evalY,total_loss,accuracy,batch_size=100):
	number_examples = len(evalX)
	eval_loss,eval_acc,eval_counter = .0,.0,.0
	for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples+1,batch_size)):
		feed_dict = {input_x:evalX[start:end],input_y:evalY[start:end],is_training:False,keep_prob:1.0}
		curr_eval_loss,curr_eval_acc = sess.run([total_loss,accuracy],feed_dict=feed_dict)
		eval_loss,eval_acc,eval_counter = eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
	return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

def unpickle(file):
	import pickle
	with open(file,'rb') as fo:
		dict = pickle.load(fo,encoding='bytes')
	return dict

def load_cifar10():
	# 导入cifar数据集，包括train_data和test_data,数据规模都为(batch,height,width,channal)
	# 数据类型为ndarray,数据标签train_labels和test_labels是一维array，范围是0-9
	height,width = 32,32
	channal = 3
	train_data = np.zeros(shape=(0,height*width*channal))
	train_labels = np.zeros(shape=0)
	# for i in range(5):
	for i in range(5):
		file_train = '../../cifar10/data_batch_' + str(i+1)
		train_dict = unpickle(file_train)
		train_data = np.vstack((train_data,train_dict[b'data']))
		train_labels = np.hstack((train_labels,np.array(train_dict[b'labels'])))
	print('end load train data.')
	n_train = train_data.shape[0]
	train_data = np.reshape(train_data,[n_train,3,height,width])
	train_data = np.transpose(train_data,(0,2,3,1))
	file_test = '../../cifar10/test_batch'
	test_dict = unpickle(file_test)
	test_data = test_dict[b'data']
	test_labels = test_dict[b'labels']
	n_test = test_data.shape[0]
	test_data =np.reshape(test_data,[n_test,3,height,width])
	test_data = np.transpose(test_data,(0,2,3,1))
	return height,width,channal,train_data,train_labels,test_data,test_labels

def load_cifar100():
	# 导入cifar数据集，包括train_data和test_data,数据规模都为(batch,height,width,channal)
	# 数据类型为ndarray,数据标签train_labels和test_labels是一维array，范围是0-9
	height,width = 32,32
	channal = 3
	train_data = np.zeros(shape=(0,height*width*channal))
	train_labels = np.zeros(shape=0)
	file_train = '../../cifar100/train'
	train_dict = unpickle(file_train)
	train_data = train_dict[b'data']
	train_labels = train_dict[b'fine_labels']
	train_labels = np.array(train_labels)
	print('end load train data.')
	n_train = train_data.shape[0]
	train_data = np.reshape(train_data,[n_train,3,height,width])
	train_data = np.transpose(train_data,(0,2,3,1))
	file_test = '../../cifar100/test'
	test_dict = unpickle(file_test)
	test_data = test_dict[b'data']
	test_labels = test_dict[b'fine_labels']
	n_test = test_data.shape[0]
	test_data =np.reshape(test_data,[n_test,3,height,width])
	test_data = np.transpose(test_data,(0,2,3,1))
	return height,width,channal,train_data,train_labels,test_data,test_labels

def pre_treatment(train_data,train_labels,test_data,test_labels,valid_portion):
	n_train = len(train_data)
	trainX = train_data
	trainY = train_labels
	# validX = train_data[int((1-valid_portion)*n_train):]
	# validY = train_labels[int((1-valid_portion)*n_train):]
	testX = test_data
	testY = test_labels
	return trainX,trainY,testX,testY

def data_normalization(train_data_raw, test_data_raw, normalize_type):
	if normalize_type == 'divide-255':
		train_data = train_data_raw/255.0
		test_data = test_data_raw/255.0
		images = np.concatenate((train_data,test_data),axis=0)
		pixel_mean = np.mean(images,axis=0)
		pixel_std = np.std(images,axis=0)
		train_data = (train_data-pixel_mean)/pixel_std
		test_data = (test_data-pixel_mean)/pixel_std
		return train_data, test_data

	elif normalize_type=='divide-256':
		train_data=train_data_raw/256.0
		test_data=test_data_raw/256.0

		return train_data, test_data
	elif normalize_type=='by-channels':
		train_data=np.zeros(train_data_raw.shape)
		test_data=np.zeros(test_data_raw.shape)
		for channel in range(train_data_raw.shape[-1]):
			images=np.concatenate((train_data_raw, test_data_raw), axis=0)
			channel_mean=np.mean(images[:,:,:,channel])
			channel_std=np.std(images[:,:,:,channel])
			train_data[:,:,:,channel]=(train_data_raw[:,:,:,channel]-channel_mean)/channel_std
			test_data[:,:,:,channel]=(test_data_raw[:,:,:,channel]-channel_mean)/channel_std

		return train_data, test_data

	elif normalize_type=='None':

		return train_data_raw, test_data_raw

def myshuffle(trainX,trainY):
	n_train = len(trainX)
	index = list(range(n_train))
	random.shuffle(index)
	trainX = trainX[index]
	trainY = trainY[index]
	return trainX,trainY

def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]
    return cropped

def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images

def count_params():
    total_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        params=1
        for dim in shape:
            params=params*dim.value
        total_params+=params
    print("Total training params: %.2fM" % (total_params / 1e6))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--aug',default=True,type=bool)
	parser.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100'])
	parser.add_argument('--n_value',default=1,type=int)
	parser.add_argument('--normalize_type',default='by-channels',choices=['by-channels','divide-255','divide-256'])
	parser.add_argument('--gpu',default=0,choices=[0,1])
	parser.add_argument('--keep_prob',default=0.8)
	args = parser.parse_args()

	gpu_id = args.gpu
	CUDA_VISIBLE_DEVICES = gpu_id
	dataset = args.dataset
	if_aug = args.aug
	my_keep_prob = args.keep_prob 
	name = os.path.basename(sys.argv[0]).split(".")[0]


	n_value = args.n_value
	normalize_type = args.normalize_type 
	if dataset == 'cifar10':
		height,width,channal,train_data,train_labels,test_data,test_labels = load_cifar10()
		num_classes = 10
	elif dataset == 'cifar100':
		height,width,channal,train_data,train_labels,test_data,test_labels = load_cifar100()
		num_classes = 100

	ckpt_dir = name + '_checkpoint/'

	if os.path.exists(ckpt_dir) == False:
		os.mkdir(ckpt_dir)

	train_data, test_data = data_normalization(train_data,test_data,normalize_type)
	valid_portion = .1
	trainX,trainY,testX,testY = pre_treatment(train_data,train_labels,test_data,test_labels,valid_portion)
	print('training dataset: %d' %(len(trainX)))
	# print('valid dataset: %d' %(len(validX)))
	print('test dataset: %d' %(len(testX)))

	print('end process data.')

	# 构建计算图
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# 导入预训练的结果
		origin_learning_rate = .1
		num_epoches = 300
		validate_every = 1
		save_every = 1
		batch_size = 128
		best_acc = 0.0

		learning_rate,input_x,input_y,is_training,keep_prob,epoch_step,epoch_increment,iteration_step,iteration_increment,total_loss,train_op,accuracy \
		= resnet_v2_simple(num_classes,batch_size,height,width,channal,n=n_value)

		saver = tf.train.Saver()
		
		if os.path.exists(ckpt_dir+'checkpoint'):
			print('Restoring Variables from checkpoint')
			saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
		else:
			print('Initializing Variables')
			sess.run(tf.global_variables_initializer())

		if os.path.exists(name + '_result'):
			record_train_loss = np.load(name + '_result/train_loss.npy')
			record_train_acc = np.load(name + '_result/train_acc.npy')
			record_test_loss = np.load(name + '_result/test_loss.npy')
			record_test_acc = np.load(name + '_result/test_acc.npy')

		else:
			os.mkdir(name+'_result')
			record_train_loss = np.zeros(num_epoches)
			record_train_acc = np.zeros(num_epoches)
			record_test_loss = np.zeros(num_epoches)
			record_test_acc = np.zeros(num_epoches)
			np.save(name + '_result/train_loss.npy',record_train_loss)
			np.save(name + '_result/train_acc.npy',record_train_acc)
			np.save(name + '_result/test_loss.npy',record_test_loss)
			np.save(name + '_result/test_acc.npy',record_test_acc)

		count_params()
		curr_epoch = sess.run(epoch_step)

		# 输入数据并训练
		number_of_training_data = len(trainX)
		for epoch in range(curr_epoch,num_epoches):
			# iteration = sess.run(iteration_step)

			# 	break
			# if epoch <= 1:
			# 	cur_learning_rate = .01
			if epoch < 150:
				cur_learning_rate = origin_learning_rate
			elif epoch < 225:
				cur_learning_rate = origin_learning_rate * .1
			elif epoch < 300:
				cur_learning_rate = origin_learning_rate * .01
			# resnet_arg_scope(is_training=True)
			loss, acc, counter = .0, .0, 0
			trainX, trainY = myshuffle(trainX,trainY)
			if if_aug:
				cur_trainX = augment_all_images(trainX,pad=4) 
			else:
				cur_trainX = trainX
			for start,end in zip(range(0,number_of_training_data,batch_size),range(batch_size,number_of_training_data+1,batch_size)):
				feed_dict = {input_x:cur_trainX[start:end],input_y:trainY[start:end],is_training:True,learning_rate:cur_learning_rate,keep_prob:my_keep_prob}
				curr_loss,curr_acc,_ = sess.run([total_loss,accuracy,train_op],feed_dict)
				sess.run(iteration_increment)
				loss,counter,acc = loss+curr_loss,counter+1,acc+curr_acc
				display_num = 50
				if counter % display_num == 0:
					print('Epoch %d\t Iteration %d\t Train Loss:%.3e\t Train Accuracy:%.5f' %(epoch,counter,loss/float(display_num),acc/float(display_num)))
					record_train_loss[epoch] = loss/float(display_num)
					record_train_acc[epoch] = acc/float(display_num)
					loss,acc =.0,.0

			sess.run(epoch_increment)
			# if epoch % validate_every == 0:
			# 	# resnet_arg_scope(is_training=False)
			# 	eval_loss,eval_acc = do_eval(sess,validX,validY,total_loss,accuracy)
			# 	print('Epoch %d\tValidation Loss:%.3e\tValidation Accuracy:%.5f' %(epoch,eval_loss,eval_acc))
				# 保存模型
			if epoch % save_every == 0:
				save_path = ckpt_dir+'model.ckpt'
				saver.save(sess,save_path,global_step=epoch)
				test_loss,test_acc = do_eval(sess,testX,testY,total_loss,accuracy)

				record_test_loss[epoch] = test_loss
				record_test_acc[epoch] = test_acc
				
				np.save(name + '_result/train_loss.npy',record_train_loss)
				np.save(name + '_result/train_acc.npy',record_train_acc)
				np.save(name + '_result/test_loss.npy',record_test_loss)
				np.save(name + '_result/test_acc.npy',record_test_acc)

				print('Epoch %d\tTest Loss:%.3e\tTest Accuracy:%.5f' %(epoch,test_loss,test_acc))
				if test_acc > best_acc:
					best_acc = test_acc
				print('The best Accuracy is %.5f' %(best_acc))






	