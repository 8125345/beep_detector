"""Write spectrograms of wav files to JSON.

Usage: onsets_frames_transcription_specjson file1.wav [file2.wav file3.wav]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from deepiano.wav2mid import configs
from deepiano.wav2mid import data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string(
    'hparams',
    'onset_mode=length_ms,onset_length=32',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def create_spec(filename, hparams):
  """Processes an audio file into a spectrogram."""
  wav_data = tf.gfile.Open(filename).read()
  spec = data.wav_to_spec(wav_data, hparams)
  return spec


def main(argv):
  tf.logging.set_verbosity(FLAGS.log)

  config = configs.CONFIG_MAP[FLAGS.config]
  hparams = config.hparams

  for filename in argv[1:]:
    tf.logging.info('Generating spectrogram for %s...', filename)

    spec = create_spec(filename, hparams)
    spec_filename = filename + '.json'
    with tf.gfile.Open(spec_filename, 'w') as f:
      f.write(json.dumps(spec.tolist()))
      tf.logging.info('Wrote spectrogram json to %s.', spec_filename)


def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
