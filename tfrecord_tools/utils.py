import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
    e.g, sentence in list of ints
    """
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
    e.g, sentence in list of bytes
    """
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def create_tfrecord(dataset_list, save_path, fun):
    """
    创建tfrecord，返回数量
    :param dataset_list: 需要打包为tfrecord的数据
    :param save_path: tfrecord路径
    :param fun: 格式化函数，用户自定义，用于预处理数据集中数据
    :return: tfrecord数量，用于验证
    """
    tfrecord_path = save_path
    total_num = 0  # 统计总数
    # 创建tfrecord
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for index, raw_line in enumerate(dataset_list):
            line = raw_line

            fun_rtn = fun(line)
            if len(fun_rtn) == 2:
                # 非固定长度序列
                feature, feature_list = fun(line)
                context = tf.train.Features(feature=feature)
                feature_lists = tf.train.FeatureLists(feature_list=feature_list)
                seq_example = tf.train.SequenceExample(
                    context=context,
                    feature_lists=feature_lists,
                )
                writer.write(seq_example.SerializeToString())
            elif len(fun_rtn):
                # 固定长度序列
                feature = fun(line)
                features = tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
            else:
                raise Exception("返回数量错误")

            total_num += 1
            if index % 1000 == 0:
                print(index)
    # TFRecordWriter tf1.15似乎存在内存泄漏
    del writer
    return total_num
