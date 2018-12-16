# Tacotron
**NOTICE: GMM HASN'T BEEN IMPLEMENTED YET.**

A fork of **[begeekmyfriend's implementation of Tacotron speech synthesis in TensorFlow](https://github.com/begeekmyfriend/tacotron)** implementing guided attention, ML-AILABS preprocessing.



## Background

In April 2017, Google published a paper, [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf),
where they present a neural text-to-speech model that learns to synthesize speech directly from
(text, audio) pairs. However, they didn't release their source code or training data. This repo is based on an
[independent attempt by Keith Ito](https://github.com/keithito/Tacotron) to provide an open-source implementation of the model described in their paper.

This paper was [initially forked by begeekmyfriend](https://github.com/begeekmyfriend/tacotron) in order to add 
Location Sensitive Attention as detailed by
[Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884).
This implementation was heavily based on [Rayhene Mamah's implementation of Tacotron2](https://github.com/Rayhane-mamah/Tacotron-2).

This fork of Tacotron implements guided attention as detailed in [Efficiently Trainable Text-to-Speech System Based on 
Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969), which have
successfully resulted in faster training in both the Deep Convolutional Text to Speech (DC-TTS) and WaveNet 
architectures, with higher quality audio being produced by the resulting
models. The code referenced was [Kyubyong's implementation of DC-TTS](https://github.com/Kyubyong/dc_tts).

## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```


### Using a pre-trained model

1. **Download and unpack a model**:
   ```
   curl http://data.keithito.com/data/speech/tacotron-20170720.tar.bz2 | tar xjC /tmp
   ```

2. **Run the demo server**:
   ```
   python3 demo_server.py --checkpoint /tmp/tacotron-20170720/model.ckpt
   ```

3. **Point your browser at localhost:9000**
   * Type what you want to synthesize



### Training

*Note: you need at least 40GB of free disk space to train a model.*

1. **Download a speech dataset.**

   The following are supported out of the box:
    * [M-AILABs en_UK](http://m-ailabs.bayern/en/the-mailabs-speech-dataset/) (Public Domain)
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
    * [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)

   You can use other datasets if you convert them to the right format. See [TRAINING_DATA.md](TRAINING_DATA.md) for more info.


2. **Unpack the dataset into `~/tacotron`**

   After unpacking, your tree should look like this for M-AILABS en_uk:
   ```
   tacotron
     |- en_UK
         |- by_book
             |- female
                 |- elizabeth_klett
                     |- wives_and_daughters
                         |- wavs
                         |- metadata.csv
                     |- jane_eyre
                         |- wavs
                         |- metadata.csv
   ```

3. **Preprocess the data**
   ```
   python3 preprocess.py --dataset en_UK
   ```

4. **Train a model**
   ```
   python3 train.py
   ```

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command
   line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"`.
   Hyperparameters should generally be set to the same values at both training and eval time.
   The default hyperparameters are recommended for LJ Speech and other English-language data.
   See [TRAINING_DATA.md](TRAINING_DATA.md) for other languages.


5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`.

6. **Synthesize from a checkpoint**
   ```
   python3 demo_server.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   Replace "185000" with the checkpoint number that you want to use, then open a browser
   to `localhost:9000` and type what you want to speak. Alternately, you can
   run [eval.py](eval.py) at the command line:
   ```
   python3 eval.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   If you set the `--hparams` flag when training, set the same value here.

## Notes
### Notes and Common Issues

  * [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) seems to improve
    training speed and avoids occasional slowdowns seen with the default allocator. You
    can enable it by installing it and setting `LD_PRELOAD=/usr/lib/libtcmalloc.so`. With TCMalloc,
    you can get around 1.1 sec/step on a GTX 1080Ti.

  * You can train with [CMUDict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) by downloading the
    dictionary to ~/tacotron/training and then passing the flag `--hparams="use_cmudict=True"` to
    train.py. This will allow you to pass ARPAbet phonemes enclosed in curly braces at eval
    time to force a particular pronunciation, e.g. `Turn left on {HH AW1 S S T AH0 N} Street.`

  * If you pass a Slack incoming webhook URL as the `--slack_url` flag to train.py, it will send
    you progress updates every 1000 steps.
    
  * During eval and training, audio length is limited to `max_iters * outputs_per_step * frame_shift_ms`
    milliseconds. With the defaults (max_iters=200, outputs_per_step=5, frame_shift_ms=12.5), this is
    12.5 seconds.
    
    If your training examples are longer, you will see an error like this:
    `Incompatible shapes: [32,1340,80] vs. [32,1000,80]`
    
    To fix this, you can set a larger value of `max_iters` by passing `--hparams="max_iters=300"` to
    train.py (replace "300" with a value based on how long your audio is and the formula above).
### TODO
  * Implementation of LPCNet vocoder instead of Griffin-Lim.
  
  * A model successfully trained with 6 hours of the M-AILABs dataset, implementing guided attention.

## Related projects
### Tacotron
#### Tacotron
  * **Initial repo by Keith Ito: https://github.com/keithito/Tacotron**
    * **Fork by begeekmyfriend: https://github.com/begeekmyfriend/Tacotron**
  * By Alex Barron: https://github.com/barronalex/Tacotron
  * By Kyubyong Park: https://github.com/Kyubyong/tacotron
#### Tacotron2
  * **Referenced Tacotron2 repo by Rayhene Mamah: https://github.com/Rayhane-mamah/Tacotron-2**
  * By NvIDIA: https://github.com/NVIDIA/tacotron2
### Misc
  * **Referenced DC-TTS repo by Kyubyong: https://github.com/Kyubyong/dc_tts/**
  * Mozilla/TTS by Mozilla: https://github.com/mozilla/TTS
  * LPCNet by Jean-Marc Valin: https://github.com/mozilla/LPCNet
  * nv-wavenet by NvIDIA: https://github.com/NVIDIA/nv-wavenet
  * Wavenet Vocoder by r9y9: https://github.com/r9y9/wavenet_vocoder
  