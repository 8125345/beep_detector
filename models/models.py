import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from tensorflow.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
    Dropout, MaxPooling2D, Concatenate, Lambda, Reshape, LayerNormalization, Resizing, AveragePooling2D
from tensorflow.keras.models import Model
from keras import backend
from keras.applications import imagenet_utils

from keras.activations import softmax
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K



# from models.metrics import cAUC, cPrecision, cRecall


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])


def conv_bn_act(x, filters, kernel_size, stride, activation, block_id):
    prefix = 'conv_bn_act_{}/'.format(block_id)
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        use_bias=False,
        name=prefix + 'expand')(
        x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand/BatchNorm')(
        x)
    if activation is not None:
        x = activation(x)
    return x


def bilstm_decoder(x, units=256):
    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units, return_sequences=True), name="cudnn_lstm"
    )(x)

    return x


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # 确保永远是divisor的倍数
    # 避免求整后卷积和数量比预期低10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(
        x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(
        x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(
            x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(
            x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + 'depthwise/pad')(
            x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
        x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(
        x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


def dw_block(x, filters, stride, activation, block_id, last_act=True):
    shortcut = x
    prefix = 'dw_block_{}/'.format(block_id)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'dw')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'dw/BatchNorm')(x)

    x = activation(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'pw/Conv')(
        x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'pw/BatchNorm')(
        x)
    if last_act:
        x = activation(x)
    return x


def dual_crnn(input_shape, input_channel, alpha=0.5, save_path=None):
    """
    输入长度为None，但是要保证整除倍数
    :param input_shape:
    :param alpha:
    :return:
    """

    assert input_channel in [1]

    def depth(d):
        return _depth(d * alpha)

    # 端到端-base
    def full_chunk_fn(x, kernel, activation, se_ratio):
        # x (32, 240, 2)
        _, _, _, ch = backend.int_shape(x)
        assert ch == input_channel

        # -----------------------------------------
        # detail branch

        x8 = x
        # detail stage1
        x8 = conv_bn_act(x8, 16, 3, (1, 2), relu, "detail_s1_1")  # 8 120
        x8 = conv_bn_act(x8, 16, 3, (1, 1), relu, "detail_s1_2")  # 8 120
        # detail stage2
        x8 = conv_bn_act(x8, depth(48), 3, (1, 2), activation, "detail_s2_1")  # 8 60
        x8 = conv_bn_act(x8, depth(48), 3, (1, 1), activation, "detail_s2_2")  # 8 60
        # detail stage3
        x8 = conv_bn_act(x8, depth(48), 3, (1, 1), activation, "detail_s3_1")  # 8 60
        # x8 = conv_bn_act(x8, depth(48), 3, (1, 1), activation, "detail_s3_2")  # 8 60

        # -----------------------------------------
        # diff branch

        # stem
        # 首次下采样
        x = conv_bn_act(x, 16, 3, 2, activation, "diff_stem")  # 4 120
        stem_conv_redu = conv_bn_act(x, 8, 1, 1, activation, "diff_stem_redu")  # 4 120
        stem_conv_redu = conv_bn_act(stem_conv_redu, 16, 3, 2, activation, "diff_stem_redu1")  # 2 60
        stem_mp = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="stem_mp")(x)  # 2 60
        x = Concatenate()([stem_conv_redu, stem_mp])  # 2 60

        # backbone
        # x = _inverted_res_block(x, 1, depth(24), 3, 1, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72. / 16, depth(24), 3, 1, None, relu, 1)
        x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)

        x = MaxPooling2D((3, 3), strides=(1, 2), padding="same")(x)  # 2 30
        x = _inverted_res_block(x, 4, depth(40), kernel, 1, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
        x = MaxPooling2D((3, 3), strides=(1, 2), padding="same")(x)  # 2 15
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 8)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 9)

        # -----------------------------------------
        # Aggregation
        x8_b1 = dw_block(x8, depth(48), 1, activation, "x8_b1_1", last_act=False)
        x8_b2 = conv_bn_act(x8, depth(48), 3, 2, None, "x8_b2")
        x8_b2 = AveragePooling2D((3, 3), strides=(2, 2), padding="same")(x8_b2)

        x_b1 = dw_block(x, depth(48), 1, hard_sigmoid, "x_b1", last_act=True)
        x_b2 = conv_bn_act(x, depth(48), 3, 1, None, "x_b2")
        x_b2 = UpSampling2D((4, 4))(x_b2)
        x_b2 = hard_sigmoid(x_b2)

        x8_f = Multiply()([x8_b1, x_b2])
        x_f = Multiply()([x8_b2, x_b1])
        x_f = UpSampling2D((4, 4))(x_f)
        x = Add()([x8_f, x_f])

        x = conv_bn_act(x, depth(48), 3, 1, activation, "x_out1")
        x = conv_bn_act(x, depth(48), 3, 1, activation, "x_out2")

        return x

    # 配置参数，按照cpu推理设置参数
    # from keras.applications.mobilenet_v3 import
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25
    inputs = keras.Input(shape=input_shape)
    rx = Resizing(8, 240)(inputs)  # 调整尺寸，便于整数倍下采样

    feature = full_chunk_fn(rx, kernel, activation, se_ratio)
    _, row, col, ch = backend.int_shape(feature)

    new_shape = (row, col * ch)
    x = keras.layers.Reshape(target_shape=new_shape, name="flatten_end")(feature)
    x = keras.layers.Dense(768, activation="relu", name="fc_end")(x)  # 对特征进行下采样
    # lstm_output = bilstm_decoder(x, units=256 + 128)
    lstm_output = bilstm_decoder(x, units=256)

    output_midi = keras.layers.Dense(
        1, activation="sigmoid", name="onset_probs"
    )(lstm_output)

    # output_midi = feature

    model = Model(inputs, outputs=[output_midi], name=f"dual_crnn_{alpha}")
    if save_path is not None:
        print(f"模型保存路径{save_path}")
        model.save(save_path)
    return model


def light_crnn(input_shape, save_path=None):
    # 所有特征不定长
    feature_input = keras.Input(shape=input_shape)

    kernel_size = 5
    activation = relu
    stride = (1, 2)

    x = feature_input
    x = conv_bn_act(x, 48, kernel_size, stride, activation, 0)
    x = conv_bn_act(x, 72, kernel_size, stride, activation, 1)
    x = conv_bn_act(x, 98, kernel_size, stride, activation, 2)

    _, row, col, ch = backend.int_shape(x)
    new_shape = (-1, col * ch)
    x = keras.layers.Reshape(target_shape=new_shape, name="flatten_end")(x)
    x = keras.layers.Dense(512, activation="relu", name="fc_end")(x)  # 对特征进行下采样
    lstm_output = bilstm_decoder(x, units=256)
    beep_prob = keras.layers.Dense(
        1, activation="sigmoid", name="beep_prob"
    )(lstm_output)

    model = Model(inputs=[feature_input], outputs=[beep_prob],
                  name=f"light_crnn")

    if save_path is not None:
        print(f"模型保存路径{save_path}")
        model.save(save_path)
    return model


def weighted_bce(weight):
    """
    加权交叉熵,注意output必须经过sigmoid，即其范围在[0, 1]
    :param weight:
    :return:
    """
    _EPSILON = 1e-7

    def loss(target, output):
        # 注意，不要裸写交叉熵，否则很难保证数值稳定性
        # https://github.com/keras-team/keras/blob/5a7a789ee9766b6a594bd4be8b9edb34e71d6500/keras/backend/tensorflow_backend.py#L3275
        # if not from_logits transform back to logits
        _epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))

        # return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
        #                                                logits=output)
        # https://tensorflow.google.cn/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        # return tf.nn.weighted_cross_entropy_with_logits(labels=target,
        #                                                 logits=output,
        #                                                 pos_weight=weight,
        #                                                 name=None)

        # 按照某个轴，对张量求平均值
        return tf.math.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=target,
                logits=output,
                pos_weight=weight,
                name=None))

    return loss


def custom_load_model(pretrain_model, lr=0.0001, compile_pretrain=False, pos_weight=1,
                      rec_loss_fun="weighted_bce", model_structure="light_crnn",
                      first_decay_steps=20000):
    """
    编译模型
    :param rec_loss_fun: 声音识别损失函数类型
    :param lr: 初始学习率
    :param compile_pretrain: 重新编译模型，主要用于保留之前训练配置
    :param pretrain_model: 预训练模型路径，或None
    :return:
    """
    if rec_loss_fun == "weighted_bce":
        # 正样本加权bce
        rec_loss = weighted_bce(pos_weight)
    elif rec_loss_fun == "bce":
        # bce
        rec_loss = "binary_crossentropy"
    else:
        raise Exception(f"损失函数：{rec_loss_fun}不存在")

    # 为了统一加载形式，确保每次训练都使用预训练模型，如果是第一次训练，提前保存一个随机参数的预训练模型
    assert pretrain_model is not None
    print(f"使用预训练模型:{pretrain_model}")
    # 针对不同的模型结构使用不同的编译方法

    model = load_model(pretrain_model, compile=False)
    lr_decayed_fn = (tf.keras.optimizers.schedules.CosineDecayRestarts(lr, first_decay_steps))

    adam = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)
    opt = adam

    loss = rec_loss
    loss_weights = 1

    # label_range = None  # (0, 8)  # 在多任务学习的时候，label需要对label长度进行裁剪，[0, 8)
    # metric_range = (3, 5)  # 计算评估指标的区间
    #
    # metrics = [
    #     # 指定范围
    #     cAUC(label_range=label_range, metric_range=metric_range),
    #     cPrecision(label_range=label_range, metric_range=metric_range),
    #     cRecall(label_range=label_range, metric_range=metric_range),
    #     cRecall(thresholds=0.1, label_range=label_range, metric_range=metric_range)
    # ]
    metrics = [
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Recall(thresholds=0.1),
    ]

    model.compile(
        optimizer=opt,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    model.summary()
    return model


if __name__ == '__main__':
    # 编译模型
    # spliter_detector
    pretrain_model = "/data/projects/LabelModels/spliter_detector/train_output_models/light_crnn.h5"
    # pretrain_model = None
    # 直接创建模型
    model = light_crnn(
        input_shape=(None, 229, 1),
        save_path=pretrain_model,
    )
    model.summary()

    import numpy as np

    test_feature = np.random.random((1, 480, 229, 1))

    out = model.predict(test_feature)

    print(out.shape)
