# TransferLearningNovelty
Author : Pramuditha Perera (pperera3@jhu.edu)

Caffe/Python Implementation of the paper Implementation of the paper <b>Learning Deep Features for One-Class Classification</b>,http://openaccess.thecvf.com/content_CVPR_2019/html/Perera_Deep_Transfer_Learning_for_Multiple_Class_Novelty_Detection_CVPR_2019_paper.html.</b>

In a practical novelty detection application, often there exists external dataset that can be used to transfer knowedge from. This work studies the problem of novelty detection in this context. We use Places365 as the external dataset.

If you found this code useful please cite our paper:

<pre><code>
@InProceedings{Perera_2019_CVPR,
author = {Perera, Pramuditha and Patel, Vishal M.},
title = {Deep Transfer Learning for Multiple Class Novelty Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
</code></pre>

Pre-processing
--------------
1. This code was developed targeting pycaffe framework with python2.7.
2. Download the code into caffe/examples folder.
3. Download pre-trained models to caffe/models folder.
	For Places365 Alexnet code visit : https://github.com/CSAILVision/places365
4. Download reference dataset to caffe/data. We used Places365 validation set. It can be found at http://places2.csail.mit.edu/download.html
5. Download target datasets to caffe/data. For novelty detection we use Caltech 256 : http://www.vision.caltech.edu/Image_Datasets/Caltech256/
6. Edit prototext files to reflect correct paths. Specifically, 
   Change Net path and snapshot_prefix in  alex/solverCal.prototxt and alex/solverCalM.prototxt
   Change "source" of "data" layer in alexCal.prototxt and  alexCalM.prototxt
7. Move APLoss.py to caffe/python folder.   


Running the Code
----------------

If dataset is stored in /home/user/data/caltech folder, run the following command:
python runOpenMulti.py --dataset /home/user/data/caltech --type deepNovelty

Baseline finetuning model can be run using:
python runOpenMulti.py --dataset /home/user/data/caltech --type mclass


Arguments
----------
1.--name : Name of the network. Used to name the performance curve plot and text output containing match scores.

2.--type : Type of CNN : deepNovelty/ mclass. Proposed method will be used when deepNovelty is specified. mclass invokes baseline finetuning. 

3.--dataset : Specify the path to the training dataset. Eg: data/caltech/

4.--cafferoot : Specify the path to the caffe installation. Default is : /home/labuser/caffe/

5.--nclasses : Number of total classes in the dataset. 256 for Caltech256.



output
------
A text file with detection score values will be written to the output folder. A ROC
curve will also be generated.
