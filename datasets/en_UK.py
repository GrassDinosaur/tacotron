from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from hparams import hparams
from util import audio


_max_out_length = 700
_end_buffer = 0.05
_min_confidence = 90

books = [
  #'wives_and_daughters',
  'jane_eyre',
]

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1

  for book in books:
    with open(os.path.join(in_dir, 'by_book/female/elizabeth_klett', book, 'metadata.csv')) as f:
      for line in f:
        if 0 == 0:#index < 3000: #Roulette to the first 2999 examples dab XD lmao approx 6 hours
          parts = line.strip().split('|')
          wav_path = os.path.join(in_dir, 'by_book/female/elizabeth_klett', book, 'wavs', '%s.wav' % parts[0])

          if os.path.isfile(wav_path):
            text = parts[2]
            futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
            index += 1
        else:
          break
  results = [future.result() for future in tqdm(futures)]
  return [r for r in results if r is not None]


def _process_utterance(out_dir, index, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'en_uk-spec-%05d.npy' % index
  mel_filename = 'en_uk-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)
