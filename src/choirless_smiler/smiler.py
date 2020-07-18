try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import argparse
import bz2
import gzip
from pathlib import Path

import cv2

import dlib

import numpy as np

import requests

from tqdm import tqdm

from tensorflow import lite as tflite

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import imutils.video

DEFAULT_LANDMARKS = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DEFAULT_CACHE_DIR = "~/.smiler"


def main():
    parser = argparse.ArgumentParser(
        description='Save thumbnail of smiliest frame in video')
    parser.add_argument('video_fn', type=str,
                        help='filename for video to analyse')
    parser.add_argument('image_fn', type=str,
                        help='filename for output thumbnail')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose mode')
    parser.add_argument('--threshold', type=int, default=0,
                        help='threshold of difference over'
                             'which we analyse an image')
    parser.add_argument('--landmarks-url', type=str,
                        default=DEFAULT_LANDMARKS,
                        help='url of facial landmark file')
    parser.add_argument('--cache-dir', type=str,
                        default=DEFAULT_CACHE_DIR,
                        help='local cache to store the landmark file in')
    parser.add_argument('--quantile', type=float, default=0.95,
                        help='quantile of images to analyse')

    args = parser.parse_args()

    with pkg_resources.path('choirless_smiler', 'smile_detector.tflite') as model_path:

        landmarks_path = load_landmarks(args.landmarks_url, args.cache_dir)

        smiler = Smiler(landmarks_path, model_path, verbose=args.verbose)
        if args.verbose:
            smiler.total_frames = imutils.video.count_frames(args.video_fn)

        if args.threshold == 0:
            fg = smiler.frame_generator(args.video_fn)
            threshold = smiler.calc_threshold(fg, args.quantile)
        else:
            threshold = args.threshold

        if args.verbose:
            print("threshold", threshold)

        fg = smiler.frame_generator(args.video_fn)
        ffg = smiler.filter_frames(fg, threshold)

        smile_score, image = smiler.find_smiliest_frame(ffg)

        # Write out our "best" frame and clean up
        if args.verbose:
            print(f"Best smile score: {smile_score:.2f}")
        cv2.imwrite(args.image_fn, image)


def load_landmarks(landmarks_url, cache_dir):

    cache_dir = Path(cache_dir).expanduser()
    if not cache_dir.exists():
        cache_dir.mkdir()

    landmarks_path = cache_dir.joinpath('landmarks.dat')
    if not landmarks_path.exists():
        req = requests.get(landmarks_url)
        with open(landmarks_path, 'wb') as f:
            if landmarks_url.endswith('.bz2'):
                f.write(bz2.decompress(req.content))
            elif landmarks_url.endswith('.gz'):
                f.write(gzip.decompress(req.content))
            else:
                f.write(req.content)

    return landmarks_path


class Smiler():
    def __init__(self, landmarks_path, model_path, verbose=False):

        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(landmarks_path))
        self.face_aligner = FaceAligner(self.predictor, desiredFaceWidth=256)
        self.verbose = verbose
        self.total_frames = None

    def frame_generator(self, video_fn):
        cap = cv2.VideoCapture(video_fn)

        while 1:
            # Read each frame of the video
            ret, frame = cap.read()

            # End of file, so break loop
            if not ret:
                break

            yield frame

        cap.release()

    def calc_threshold(self, frames, q=0.95):
        prev_frame = next(frames)
        counts = []

        if self.verbose:
            if self.total_frames is not None:
                frames = tqdm(frames, total=self.total_frames)
            else:
                frames = tqdm(frames)
            frames.set_description("Calculating threshold")

        for frame in frames:
            # Calculate the pixel difference between the current
            # frame and the previous one
            diff = cv2.absdiff(frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)

            # Append the count to our list of counts
            counts.append(non_zero_count)
            prev_frame = frame

        self.total_frames = int(self.total_frames * (1 - q))
        return int(np.quantile(counts, q))

    def filter_frames(self, frames, threshold):
        prev_frame = next(frames)
        for frame in frames:
            # Calculate the pixel difference between the current
            # frame and the previous one
            diff = cv2.absdiff(frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)

            if non_zero_count > threshold:
                yield frame

            prev_frame = frame

    def find_smiliest_frame(self, frames, callback=None):

        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        def detect(gray, frame):
            # detect faces within the greyscale version of the frame
            faces = self.detector(gray, 2)
            smile_score = 0

            # For each face we find...
            for rect in faces:
                (x, y, w, h) = rect_to_bb(rect)
                face_orig = imutils.resize(frame[y:y + h, x:x + w], width=256)
                face_aligned = self.face_aligner.align(frame, gray, rect)

                face_aligned = face_aligned.reshape(1, 256, 256, 3)
                face_aligned = face_aligned.astype(np.float32) / 255.0

                self.interpreter.set_tensor(input_details[0]['index'],
                                            face_aligned)
                self.interpreter.invoke()
                pred = self.interpreter.get_tensor(
                    output_details[0]['index'])[0][0]

                smile_score += pred

            return smile_score, frame

        best_smile_score = 0
        best_frame = next(frames)

        if self.verbose:
            if self.total_frames is not None:
                frames = tqdm(frames, total=self.total_frames)
            else:
                frames = tqdm(frames)
            frames.set_description("Finding smiliest face")

        for frame in frames:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Call the detector function
            smile_score, frame = detect(gray, frame)

            # Check if we have more smiles in this frame
            # than out "best" frame
            if smile_score > best_smile_score:
                best_smile_score = smile_score
                best_frame = frame
                if callback is not None:
                    callback(best_frame, best_smile_score)
                if self.verbose:
                    tqdm.write(f"New smiliest score: {best_smile_score:.2f}")

        return best_smile_score, best_frame


if __name__ == '__main__':
    main()
