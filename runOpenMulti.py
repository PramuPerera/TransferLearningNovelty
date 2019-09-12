from pylab import *
import caffe
import sys
import os
from random import shuffle
import writeFileNames
import classifyImage
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import argparse

def arguments():
    """Parse arguments into a parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="deepNoveltyAlex", help="Name of the network")
    parser.add_argument("--type", default="mclass", help="Type of CNN : deepNovelty / mclass")
    parser.add_argument("--nclasses", default=256,
                        help="number of classes in dataset")
    parser.add_argument("--dataset", default="data/caltech/",
                        help="Specify the path to the training dataset")
    parser.add_argument("--cafferoot", default="/home/labuser/caffe/",
                        help="Specify the path to the caffe instalation")
    return(parser)


parser = arguments()
physical_dir = os.path.dirname(os.path.realpath(__file__))
args = parser.parse_args()
caffe_root = args.cafferoot
sys.path.insert(0, caffe_root + 'python')
caffe.set_device(0)
caffe.set_mode_gpu()
os.chdir(caffe_root)
subpath = args.dataset
path = caffe_root+subpath


def run(args):
	os.chdir(caffe_root)
	solver = None  
	if args.type == "deepNovelty" :
		solver = caffe.SGDSolver(physical_dir+'/alex/solverCalM.prototxt')
		solver.net.copy_from('models/alexnet_places365.caffemodel')

	
	elif args.type == "mclass":
		solver = caffe.SGDSolver(physical_dir+'/alex/solverCal.prototxt')
		solver.net.copy_from('models/alexnet_places365.caffemodel')


	
	niter = 20000
	test_interval = 500
	for it in range(niter):
    		solver.step(1)  
    		solver.test_nets[0].forward(start='conv1')
	os.chdir(physical_dir)
	model_def = 'deploy_alex.prototxt'
	if args.type == "deepNovelty": 
		model_weights = 'models/FTAlexCalM'+'_iter_'+str(niter)+'.caffemodel'
	elif args.type == "mclass":
		model_weights = 'models/FTAlexCal'+'_iter_'+str(niter)+'.caffemodel'



	fpr,tpr,roc_auc = classifyImage.getResultsM(model_def,model_weights,physical_dir+"/outputs"+".txt",40,'Alex',caffe_root,physical_dir+'/test.txt' )
	return fpr,tpr,roc_auc



users = range(1,int(args.nclasses)) 
user_no = int(args.nclasses/2)
os.chdir(caffe_root)
writeFileNames.write(args.dataset, users, user_no, caffe_root, physical_dir, args.type )
fpr,tpr,roc_auc = run(args)
print('AUC of ROC curve : ' + str(roc_auc))
		






