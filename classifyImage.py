def getFeature (imagepath,transformer, net):
	image = caffe.io.load_image(imagepath)

	net.blobs['data'].data[...] = image
	output = net.forward()
	feature= output['fc7'][0]  # the output probability vector for the first image in the batch
	return feature


def getFeaturesFromMatrix (matrix,transformer, net):
	net.blobs['data'].data[...] = matrix
	output = net.forward()
	feature= output['fc7']
	return feature


def getFeaturesFromMatrixM (matrix,transformer, net):
	import numpy as np
	net.blobs['data'].data[...] = matrix
	output = net.forward()
	feature= output['fc9'] 
	#feature = feature[:,0:128] #128
	return np.max(feature,1), np.argmax(feature,1)


def getFeaturesFromMatrixBi (matrix,transformer, net):
	import numpy as np
	net.blobs['data'].data[...] = matrix
	output = net.forward()
	feature= output['fc8_']
	feature = feature[:,-1]
	return feature


def getNet (model_def ,model_weights, csize, caffepath, bs):
	import caffe
	import numpy as np
	net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  	# contains the trained weights
                caffe.TEST)     	# use test mode (e.g., don't perform dropout)
	net.blobs['data'].reshape(bs,        # batch size
                          3,         	     # 3-channel (BGR) images
                          csize, csize)  	

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	
	mu = np.load(caffepath + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)  
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
	return net, transformer



def getResultsM(model_def,model_weights,outfile,bsize,nw,caffe_root,testfile):
	print(testfile)
	from sklearn.metrics import roc_curve, auc
	import numpy as np
	import matplotlib.pyplot as plt
	from random import shuffle
	import os
	import sys
	sys.path.insert(0, caffe_root + 'python')
	import caffe
	caffepath = caffe_root
	caffe.set_device(0)
	caffe.set_mode_gpu()
	if nw == 'VGG':
		loadsz = bsize
		csize = 224
	else:
		loadsz = bsize
		csize = 227

	#Get validation data set

	net, transformer = getNet (model_def ,model_weights, csize, caffepath, loadsz);	
	max_sig_size =bsize;
	images = np.empty((max_sig_size,3,csize,csize))
	nblocks = 0;
	text_file = open(str(outfile), "w")
	text_file.close()
	f = open(testfile)
	testfiles=[];
	labels=[];
	for l in f.readlines():
  	 currFileNames = l.strip().split(" " );  
  	 testfiles.append(currFileNames[0])
  	 labels.append(int(currFileNames[1])) # matched
	f.close()
	matched=[]
	lbls = []
	tot = 0
	acc = 0
	cor = 0
	#Testing
	print("TESTING..")
	max_test_size =bsize;
	count=0;
	probeimages = np.zeros((max_test_size,3,csize,csize))
	net, transformer = getNet (model_def ,model_weights,  csize, caffepath, loadsz);
	for n in testfiles:
		image = caffe.io.load_image(caffepath+n)	
   		transformed_image = transformer.preprocess('data', image)
   		probeimages[count,:,:,:] = transformed_image;
		count = count+1
		if count==max_test_size:
			#batch is complete
			count = 0;
			probefeatures, preds = getFeaturesFromMatrixM (probeimages,transformer, net)		
			probeimages = np.zeros((max_test_size,3,csize,csize))
			text_file = open(str(outfile), "a")
			for x in range(len(probefeatures)):
				matched.append(-1*probefeatures[x])
				lbls.append(labels[x+tot*bsize])
				text_file.write("%s %s\n" % (str(probefeatures[x]), str(labels[x+tot*bsize])))
				if labels[x+tot*bsize] != -1:
					acc += 1
					if labels[x+tot*bsize]  == preds[x]:
						cor+=1

			text_file.close()
			tot=tot+1
	print(cor/float(acc))
	lbls = np.array(lbls)
	matched = np.array(matched)
	lbls[lbls!=-1] = 1
	lbls[lbls==-1]= 0
	fpr, tpr, _ = roc_curve(lbls,matched, 0)
	roc_auc = auc(fpr, tpr)
	print('Area under the curve: ' + str(roc_auc))
	fig = plt.figure()
	plt.plot(fpr, tpr,lw=2, label='ROC curce ' + str(roc_auc))
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('out.png') 
	plt.close("all")
	return(fpr,tpr,roc_auc)







def getResultsBi(model_def,model_weights,outfile,bsize,nw,caffe_root):
	from sklearn.metrics import roc_curve, auc
	import numpy as np
	from random import shuffle
	import os
	import sys
	sys.path.insert(0, caffe_root + 'python')
	import caffe
	caffepath = caffe_root
	caffe.set_device(0)
	caffe.set_mode_gpu()
	if nw == 'VGG':
		loadsz = bsize
		csize = 224
	else:
		loadsz = bsize
		csize = 227

	#Get validation data set
	

	net, transformer = getNet (model_def ,model_weights, csize, caffepath, loadsz);	
	max_sig_size =bsize;
	images = np.empty((max_sig_size,3,csize,csize))
	nblocks = 0;
	text_file = open(str(outfile), "w")
	text_file.close()
	f = open("test.txt")
	testfiles=[];
	labels=[];
	for l in f.readlines():
  	 currFileNames = l.strip().split(" " );  
  	 testfiles.append(currFileNames[0])
  	 labels.append(int(currFileNames[1])) # matched
	f.close()
	matched=[];
	tot = 0
	#Testing
	print("TESTING..")
	max_test_size =bsize;
	count=0;
	probeimages = np.zeros((max_test_size,3,csize,csize))
	net, transformer = getNet (model_def ,model_weights,  csize, caffepath, loadsz);
	for n in testfiles:
		print tot
		image = caffe.io.load_image(caffepath+n)	
   		transformed_image = transformer.preprocess('data', image)
   		probeimages[count,:,:,:] = transformed_image;
		count = count+1
		if count==max_test_size:
			#batch is complete
			count = 0;
			probefeatures = getFeaturesFromMatrixBi (probeimages,transformer, net)		
			probeimages = np.zeros((max_test_size,3,csize,csize))
			text_file = open(str(outfile), "a")
			for x in range(len(probefeatures)):
				text_file.write("%s %s\n" % (str(probefeatures[x]), str(labels[x+tot*bsize])))
			text_file.close()
			tot=tot+1
	


	

	#fpr, tpr, _ = roc_curve(labels, matched, 0)
	#roc_auc = auc(fpr, tpr)
	return(0,0,0)




