import caffe
import numpy as np
import math


class APLossLayer(caffe.Layer):


    def setup(self, bottom, top):
        # check input pair
        assert len(bottom) == 2,            'requires two layer.bottom'
        assert bottom[0].data.ndim == 2,    'requires blobs of one dimension data: FC feature'
        assert bottom[1].data.ndim == 1,    'requires blobs of one dimension data: label'
        assert len(top) == 1,               'requires a single layer.top'
	self.gamma = 5.0
    def reshape(self, bottom, top):

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	n = np.shape(bottom[0].data)[0]
	m = np.shape(bottom[0].data)[1]
	loss = 0
	for i in range(0,n):
		lbl = int(bottom[1].data[i])-1
		loss = loss+(1-bottom[0].data[i,lbl])**2+self.gamma*np.sum(np.power(bottom[0].data[i,:],2))/float(m-1)-self.gamma*(bottom[0].data[i,lbl]**2)/float(m-1)
        top[0].data[...] = loss/float(n)
	#print(loss/float(n))

    def backward(self, top, propagate_down, bottom):
	n = np.shape(bottom[0].data)[0]
	m = np.shape(bottom[0].data)[1]
	deri = self.gamma*bottom[0].data*(1/float(m-1));
	for i in range(0,n):
		lbl = int(bottom[1].data[i])-1
		deri[i,lbl] = deri[i,lbl]-(self.gamma*2.0/(m-1)*bottom[0].data[i,lbl]+(1-bottom[0].data[i,lbl]))
	#print(np.isnan(deri[...]/float(n)))
	bottom[0].diff[...] = 0.1*deri[...]/float(n);

