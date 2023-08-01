import math
import json
import tensorflow as tf

def load_json(path):
    with open(path, 'rt') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, 'wt') as f:
        json.dump(data, f, indent=4)

def rgb2coef(rgb):
    r, g, b = tf.split(rgb, num_or_size_splits=3, axis=-1)
    coef = tf.concat([r * g * b,
                      r * g * (1. - b),
                      r * (1. - g) * b,
                      r * (1. - g) * (1. - b),
                      (1. - r) * g * b,
                      (1. - r) * g * (1. - b),
                      (1. - r) * (1. - g) * b,
                      (1. - r) * (1. - g) * (1. - b)],
                     axis=-1)
    return coef


def coef2rgb(coef):
    c1, c2, c3, c4, c5, c6, c7, _ = tf.split(coef, num_or_size_splits=8, axis=-1)
    rgb = tf.concat([c1 + c2 + c3 + c4,
                     c1 + c2 + c5 + c6,
                     c1 + c3 + c5 + c7],
                    axis=-1)
    return rgb


def warmup_cosine_scheduler(step, learning_rate, warmup_steps, decay_steps):
    if step < warmup_steps:
        warmup_decay = step / warmup_steps
        return learning_rate * warmup_decay
    else:
        step = step - warmup_steps
        decay_step = decay_steps - warmup_steps
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / decay_step))
        return learning_rate * cosine_decay


def psnr(y_true, y_pred):
    y_true = tf.clip_by_value(0.5 * (y_true + 1.0), 0.0, 1.0)
    y_pred = tf.clip_by_value(0.5 * (y_pred + 1.0), 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim(y_true, y_pred):
    y_true = tf.clip_by_value(0.5 * (y_true + 1.0), 0.0, 1.0)
    y_pred = tf.clip_by_value(0.5 * (y_pred + 1.0), 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, warmup_steps=2500, decay_steps=100000):
        super(WarmupCosineSchedule, self).__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            warmup_decay = step / self.warmup_steps
            return self.learning_rate * warmup_decay
        else:
            step = step - self.warmup_steps
            decay_step = self.decay_steps - self.warmup_steps
            cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / decay_step))
            return self.learning_rate * cosine_decay


class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(0.5 * (y_true + 1.0), 0.0, 1.0)
        y_pred = tf.clip_by_value(0.5 * (y_pred + 1.0), 0.0, 1.0)
        self.psnr.assign_add(tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0)))
    
    def result(self):
        return self.psnr
    
    def reset_states(self):
        self.psnr.assign(0)


class SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='ssim', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(0.5 * (y_true + 1.0), 0.0, 1.0)
        y_pred = tf.clip_by_value(0.5 * (y_pred + 1.0), 0.0, 1.0)
        self.ssim.assign_add(tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)))
    
    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0)