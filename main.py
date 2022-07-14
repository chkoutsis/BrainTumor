import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

WORKSPACE_PATH = r'\BrainTumor-Master\Tensorflow\workspace'
SCRIPTS_PATH = r'\BrainTumor-Master\Tensorflow\scripts'
APIMODEL_PATH = r"\BrainTumor-Master\Tensorflow\models"
ANNOTATION_PATH = WORKSPACE_PATH +'\Annotations'
IMAGE_PATH = WORKSPACE_PATH +"\images"
MODEL_PATH = WORKSPACE_PATH +"\models"
PRETRAINED_MODEL_PATH = WORKSPACE_PATH +"\pre-trained-models"
CONFIG_PATH = MODEL_PATH +"\my_ssd_mobnet\pipeline.config"
CHECKPOINT_PATH = MODEL_PATH +"\my_ssd_mobnet"


#Create Label Map
labels = [{'name':'Tumor', 'id':1},] 

with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


#Create TF records
get_ipython().system("python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}")
get_ipython().system("python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}")


#Download TF Models Pretrained Models from Tensorflow Model Zoo
#!cd Tensorflow && git clone https://github.com/tensorflow/models


#Copy Model Config to Training Folder
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
get_ipython().system("mkdir {r'\\BrainTumor-Master\\Tensorflow\\workspace\\models\\\\'+CUSTOM_MODEL_NAME}")
get_ipython().system("copy {PRETRAINED_MODEL_PATH+'\\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\\pipeline.config'} {MODEL_PATH+'\\\\'+CUSTOM_MODEL_NAME}")


#Update Config For Transfer Learning
CONFIG_PATH = MODEL_PATH+'\\'+CUSTOM_MODEL_NAME+'\pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  


pipeline_config.model.ssd.num_classes = 1
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']


config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

#Train the model
print("""python {}\\research\\object_detection\\model_main_tf2.py --model_dir={}\\{} --pipeline_config_path={}\\{}\\pipeline.config --num_train_steps=5000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))
