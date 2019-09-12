from pylab import *
import matplotlib.pyplot as plt
import caffe
import sys
import os
import glob
from random import shuffle
from shutil import copyfile

def shufffiles(filename):
	with open(filename,'r') as source:
    		data = [ (random(), line) for line in source ]
		data.sort()
	with open(filename,'w') as target:
    		for _, line in data:
        		target.write( line )

def write(path, users, user_no, sub_path, physical_dir, testtype):
	target = physical_dir+'/data/DeepNoveltyData/'
	if not os.path.exists(target):
    		os.makedirs(target)
	dirnames = glob.glob(path+'/*')
	for i,dname in zip(users, dirnames):
		lst = glob.glob(dname+'/*')
		text_file = open(path+"files"+str(i)+".txt", "w")
		for fl in lst:
		     if i<user_no:	
			if testtype == 'bi':
				text_file.write("%s\n" % (  target+str(i)+'@'+os.path.basename(fl) +" "+str(365+i) ))
			else:
				text_file.write("%s\n" % (  target+str(i)+'@'+os.path.basename(fl) +" "+str(i) ))
		     else:
			text_file.write("%s\n" % (  target+str(i)+'@'+os.path.basename(fl) +" "+str(-1) ))
		     copyfile(fl, target+str(i)+'@'+os.path.basename(fl))
		text_file.close()


	pos = []
	post= []
	neg = []
	users = range(1,user_no)
	for i in users:
		f = open(path+"files"+str(i)+".txt", "r")
		for l in f.readlines():
			pos.append(l)
			post.append(l[len(sub_path):] )
		
		f.close()
	for i in range(user_no+1, user_no*2):
		f = open(path+"files"+str(i)+".txt", "r")
		count = 0
		for l in f.readlines():
			# Sample equal number of positive and negative targets. Remove if needed.
			if count<np.max([1,len(post)/(user_no)/2]): 
				neg.append(l)
				count=count+1
		f.close()
	np.random.shuffle(pos)
	text_file = open(physical_dir +"/train_val_cal.txt", "w")
	print(physical_dir +"/train_val_cal.txt")
	for fl in pos[1:len(pos)/2]:
		text_file.write("%s" % (  fl ))

	text_file.close()
	shufffiles(physical_dir + "/train_val_cal.txt")
	text_file = open(physical_dir +"/test.txt", "w")
	for fl in pos[len(pos)/2:len(pos)/2+len(pos)]:
		text_file.write("%s" % (  fl[len(sub_path):] ))

	for fl in neg:
		text_file.write("%s" % (  fl[len(sub_path):] ))
	text_file.close()
		

