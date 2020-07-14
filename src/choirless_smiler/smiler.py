try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import numpy as np
import cv2
import argparse

from tensorflow import lite as tflite

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils

import dlib

def main():
    parser = argparse.ArgumentParser(description='Save thumbnail of smiliest frame in video')
    parser.add_argument('video_fn', type=str,
                        help='filename for video to analyse')
    parser.add_argument('image_fn', type=str,
                        help='filename for output thumbnail')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose mode')
    parser.add_argument('--threshold', type=int, default=0,
                        help='threshold of difference over which we analyse an image')
    parser.add_argument('--quantile', type=float, default=0.95,
    help='quantile of images to analyse')

    args = parser.parse_args()

    if args.threshold == 0:
        fg = frame_generator(args.video_fn)
        threshold = calc_threshold(fg, args.quantile)
    else:
        threshold = args.threshold

    if args.verbose:
        print("threshold", threshold)
    
    fg = frame_generator(args.video_fn)
    ffg = filter_frames(fg, threshold)

    smile_score, image = find_smiliest_frame(ffg)

    # Write out our "best" frame and clean up
    if args.verbose:
        print("Best smile score:", smile_score)
    cv2.imwrite(args.image_fn, image)

    
def frame_generator(video_fn):
    cap = cv2.VideoCapture(video_fn)

    while 1:
        # Read each frame of the video
        ret, frame = cap.read()

        # End of file, so break loop
        if not ret:
            break

        yield frame

    cap.release()

def calc_threshold(frames, q=0.95):
    prev_frame = next(frames)
    counts = []
    for frame in frames:
        # Calculate the pixel difference between the current
        # frame and the previous one
        diff = cv2.absdiff(frame, prev_frame)
        non_zero_count = np.count_nonzero(diff)

        # Append the count to our list of counts
        counts.append(non_zero_count)
        prev_frame = frame

    return int(np.quantile(counts, q))


def filter_frames(frames, threshold):
    prev_frame = next(frames)
    for frame in frames:
        # Calculate the pixel difference between the current
        # frame and the previous one
        diff = cv2.absdiff(frame, prev_frame)
        non_zero_count = np.count_nonzero(diff)

        if non_zero_count > threshold:
            yield frame

        prev_frame = frame

def find_smiliest_frame(frames):

    detector = dlib.get_frontal_face_detector()

    with pkg_resources.path('choirless_smiler', 'shape_predictor_68_face_landmarks.dat') as predictor_path:
        predictor = dlib.shape_predictor(str(predictor_path))
        fa = FaceAligner(predictor, desiredFaceWidth=256)

    with pkg_resources.path('choirless_smiler', 'smile_detector.tflite') as model_path:
        interpreter = tflite.Interpreter(model_path=str(model_path))

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    def detect(gray, frame):
        # detect faces within the greyscale version of the frame
        faces = detector(gray, 2)
        smile_score = 0

        # For each face we find...
        for rect in faces:
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(frame[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(frame, gray, rect)

            faceAligned = faceAligned.reshape(1, 256, 256, 3)
            faceAligned = faceAligned.astype(np.float32) / 255.0

            interpreter.set_tensor(input_details[0]['index'], faceAligned)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

            smile_score += pred

        return smile_score, frame

    best_smile_score = 0
    best_frame = next(frames)
    
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
        
    return best_smile_score, best_frame

if __name__ == '__main__':
    main()
    