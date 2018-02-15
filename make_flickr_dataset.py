import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *

vgg_model = '/home/ubuntu/src/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/ubuntu/src/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

annotation_path = './data/captions.txt'
flickr_image_path = './data/images/'
feat_path = './data/feats.npy'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

#ipdb.set_trace()

annotations = pd.read_table(annotation_path, sep='\t', header=None, engine='python', names=['image', 'caption'])
#annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x))

if not os.path.exists(feat_path):
    feats = cnn.get_features(annotations['image'].values)
    np.save(feat_path, feats)
