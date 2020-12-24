from __future__ import division

import copy

import cv2
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn.decomposition import FastICA

previous = 24


# creates a butter pass filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# creates a butter pass filter by cutting off certain frequency
# returns filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def main(path, pts_array):
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-v", "--video",
    #                 help="path to the (optional) video file")
    # ap.add_argument("-t", "--tracker", type=str, default="boosting",
    #                 help="OpenCV object tracker type")
    # args = vars(ap.parse_args())
    args = dict()
    args["tracker"] = "boosting"
    args["video"] = path
    # args["video"] = "/home/kpatel/Downloads/test2.mp4"

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    init_bb = None
    temp_temp_list = []
    temp_frame_list = []

    global frame, roiPts, all
    all = []
    # if the video path was not supplied, grab the reference to the camera
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
        args["video"] = "webcam"
    else:
        camera = cv2.VideoCapture(args["video"])
    roiPts = []
    imm = []
    length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 1
    tmp_cnt = 0
    while True:
        if len(roiPts) < 4:
            (grabbed, frame) = camera.read()
            if not grabbed:
                # print("not grabbed")
                break

            # frame = cv2.resize(frame, (640, 480))

            if init_bb is not None:
                (success, box) = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

                info = [
                    ("Tracker", args["tracker"]),
                    ("Success", "Yes" if success else "No"),
                ]

                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, 480 - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                HSVframe = copy.copy(frame)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                crop_img = hsv[y:y + h, x:x + w]
                h, s, v = cv2.split(crop_img)
                all1 = np.mean(h, axis=(0, 1))
                all.append(all1)

                FPS = 30
                Win = 30
                if len(
                        all) >= FPS * Win:  # fps = 30 frames/second, 300frames = 20 second moving window (30frames *20seconds).
                    result = []
                    all = all[-FPS * Win:]
                    window = np.asarray(all)
                    # print window.shape
                    ica = FastICA(whiten=False)
                    window = (window - np.mean(window, axis=0)) / np.std(window, axis=0)  # signal normalization)
                    window = np.reshape(window, (FPS * Win, 1))
                    S = ica.fit_transform(window)  # ICA Part
                    lowcut = 0.1
                    highcut = 0.5
                    detrend = scipy.signal.detrend(S)
                    y = butter_bandpass_filter(detrend, lowcut, highcut, FPS, order=3)
                    powerSpec = np.abs(np.fft.fft(y, axis=0)) ** 2
                    freqs = np.fft.fftfreq(FPS * Win, 1.0 / FPS)
                    MIN_HR_BPM = 6.0
                    MAX_HR_BMP = 30.0
                    MAX_HR_CHANGE = 6.0
                    SEC_PER_MIN = 60
                    maxPwrSrc = np.max(powerSpec, axis=1)
                    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
                    validPwr = maxPwrSrc[validIdx]
                    validFreqs = freqs[validIdx]
                    maxPwrIdx = np.argmax(validPwr)
                    hr = validFreqs[maxPwrIdx]
                    out6 = hr * 60
                    result.append(out6)
                    ave = np.asarray(result)
                    out6 = int(np.mean(ave))
                    global previous
                    previous = out6
                    # textFile = open('{}.txt'.format(args["video"][:-4]), 'w')
                    # textFile.write(str(out6))
                    # textFile.close()
                    temp_temp_list.append(out6)
                    temp_frame_list.append(str("frame_{}".format(counter)))

                    tao = str('%.2f' % (out6))
                    ce = 'RR: ' + tao
                    cv2.putText(HSVframe, ce, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            counter += 1
            # key = cv2.waitKey() & 0xFF
        if tmp_cnt == 0:  ## key == ord("s"):
            # INIT BB GIVES AN ARRAY OF DIAGONALLY OPPOSITE TWO POINTS OF THE RECTANGLE - the top left and bottom right.
            init_bb = tuple(pts_array)
            # init_bb = cv2.selectROI("frame", frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, init_bb)
            tmp_cnt = tmp_cnt + 1
    approx_pulse_rate = pd.Series(np.asarray(temp_temp_list))
    frame_name = pd.Series(np.asarray(temp_frame_list))

    df = pd.DataFrame({"Frame Name": frame_name, "Ground Truth": approx_pulse_rate})
    df.to_csv("{}.csv".format(args["video"][:-4]))

    return_value = None
    if len(approx_pulse_rate) != 0:
        return_value = str(round(sum(approx_pulse_rate) / len(approx_pulse_rate)))
    else:
        return_value = "0"

    camera.release()
    cv2.destroyAllWindows()
    return return_value


if __name__ == "__main__":
    main()
