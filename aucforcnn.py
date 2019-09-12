from sklearn.metrics import roc_curve, auc
import glob	
import ntpath
import numpy as np
import matplotlib.pyplot as plt
files = glob.glob("outputs.txt")
for fd in files:
	print(fd)
	f = open(fd)
	

	testfiles=np.array([]);
	labels=np.array([]);

	for l in f.readlines():		 
	  	 currFileNames = l.strip().split(" " );  
		 testfiles=np.append(testfiles,float(currFileNames[0]))
	  	 labels=np.append(labels,int(float(currFileNames[-1]))) # matched
		 #print(str(currFileNames[0])+" "+(str(currFileNames[-1])))
	f.close()	

	labels[labels!=-1] = 1
	labels[labels==-1]= 0
	fpr, tpr, _ = roc_curve(labels, -1*testfiles, 0)
	roc_auc = auc(fpr, tpr)
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

	print(roc_auc)
