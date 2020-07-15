# Smiler

This is a library and CLI tool to extract the "smiliest" of frame from a video of people.

It was developed as part of [Choirless](https://github.com/choirless) as part of
[IBM Call for code](https://callforcode.org).

## Installation

```
% pip install choirless_smiler
```

## Usage

Simple usage:
```
% smiler video.mp4 snapshot.jpg
```

![Output image of people singing](https://raw.githubusercontent.com/Choirless/smiler/master/_imgs/output.jpg "Snapshot of singers")

It will do a pre-scan to determine the 5% most changed frames from their previous frame
in order to just consider them. If you know the threshold of change you want to use you
can use that. e.g.

The first time smiler runs it will download facial landmark data and store it in `~/.smiler`
location of this data and cache directory can be specified as arguments

```
% smiler video.mp4 snapshot.jpg --threshold 480000
```

## Help

```
% smiler -h
usage: smiler [-h] [--verbose] [--threshold THRESHOLD]
              [--landmarks-url LANDMARKS_URL] [--cache-dir CACHE_DIR]
              [--quantile QUANTILE]
              video_fn image_fn

Save thumbnail of smiliest frame in video

positional arguments:
  video_fn              filename for video to analyse
  image_fn              filename for output thumbnail

optional arguments:
  -h, --help            show this help message and exit
  --verbose             verbose mode
  --threshold THRESHOLD
                        threshold of difference overwhich we analyse an image
  --landmarks-url LANDMARKS_URL
                        url of facial landmark file
  --cache-dir CACHE_DIR
                        local cache to store the landmark file in
  --quantile QUANTILE   quantile of images to analyse
```

## Use as a library
Smiler can be imported and used in a library. You are responsible
for supplying paths to the facial landmark data and model, but
help functions in the module can help.

```python
from choirless_smiler.smiler import Smiler, load_landmarks

landmarks_path = load_landmarks(landmarks_url, cache_dir)

smiler = Smiler(landmarks_path, model_path)

fg = smiler.frame_generator(video_fn)
threshold = smiler.calc_threshold(fg, quantile)
fg = smiler.frame_generator(video_fn)
ffg = smiler.filter_frames(fg, threshold)

smile_score, image = smiler.find_smiliest_frame(ffg)
```

## Re-training

There are some scripts in the `scripts` directory in the
[Github repo](https://github.com/choirless/smiler)
to generate new images and to aid manual classification
and retraining of the model.

