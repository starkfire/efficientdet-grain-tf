import tensorflow as tf
from models.research.object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def get_pipeline_config(path):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    
    return pipeline_config

def save_pipeline_config(pipeline_config, path):
    config_text = text_format.MessageToString(pipeline_config)

    with tf.compat.v1.gfile.Open(path, 'wb') as f:
        tf.compat.v1.logging.info("Writing pipeline config file to %s", path)
        f.write(config_text)


# specify path to the pre-trained EfficientDet weights
pipeline_config_path = './pretrained/efficientdet_d0_coco17_tpu-32/pipeline.config'
pipeline_config = get_pipeline_config(pipeline_config_path)


# modify parameters
pipeline_config.model.ssd.num_classes = 3

pipeline_config.train_config.batch_size = 8
pipeline_config.train_config.fine_tune_checkpoint = '/content/pretrained/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
pipeline_config.train_config.num_steps = 3000

pipeline_config.train_input_reader.label_map_path = '/content/annotations/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = '/content/train.record'

pipeline_config.eval_input_reader[0].label_map_path = '/content/annotations/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = '/content/test.record'


# since we're only training in one GPU
pipeline_config.train_config.replicas_to_aggregate = 1
pipeline_config.train_config.sync_replicas = False


# save updated parameters
save_pipeline_config(pipeline_config, './pipeline.config')