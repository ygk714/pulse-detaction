import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
import argparse

# Global variables
window_size = 300
fps=30
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def main():
    # Parsing arguments
    parser=argparse.ArgumentParser(description='Compute HR')
    parser.add_argument('inputVid',help='Input Video Path', type=str)
    parser.add_argument('outputVid',help='Output Video Path',type=str)
    parser.add_argument('--WinLen',help='Window length',type=int)
    parser.add_argument('--fps', help='Window length', type=int)
    parser.add_argument('--xParts', help='Window length', type=int)
    parser.add_argument('--yParts', help='Window length', type=int)

    args=parser.parse_args()
    vid_name, outputVid, xParts, yParts=handle_arg(args)

    # Defining variables
    H_mat = []
    S_mat = []
    I_mat = []
    prv_frame = []
    prv_gray = []
    p0 = []
    first_p0 = []
    curr_frame = []
    face_roi=[]
    face_roi_orig=[]
    curr_gray = []
    forehead_roi = []
    diff_x = 0
    diff_y = 0
    mask = np.zeros_like(prv_frame)

    # Opening the input video
    vidcap = cv2.VideoCapture(vid_name)
    count = 0
    pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)

    print('--HR Calculation--')
    print('Received input video named: %s' % vid_name)
    print('HR detection initiated')

    hr_vec = []
    while True:
        success, curr_frame = vidcap.read()
        if success:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            if count == 0:  # Reading the first frame
                # Get points to track
                face_roi, forehead_roi, face_roi_orig = select_roi(curr_frame)
                p0 = get_points_to_track(curr_gray, face_roi)
                first_p0 = p0
            else:
                # Track
                p1, st, err = cv2.calcOpticalFlowPyrLK(prv_gray, curr_gray, p0, None, **lk_params)

                # Find the difference in the roi
                diff_x, diff_y = get_movement_from_trackers(first_p0, p1)
                # Get the rgb average in the forehead roi
                H_mat.append(np.zeros(xParts*yParts))
                S_mat.append(np.zeros(xParts*yParts))
                I_mat.append(np.zeros(xParts*yParts))
                for i in range(xParts):
                    for j in range(yParts):
                        roi_frac=[forehead_roi[0]+i*(forehead_roi[1]-forehead_roi[0])/3,forehead_roi[1]-(2-i)*(forehead_roi[1]-forehead_roi[0])/3,
                                  forehead_roi[2] + j * (forehead_roi[3] - forehead_roi[2]) / 2,
                                  forehead_roi[3] - (1-j) * (forehead_roi[3] - forehead_roi[2]) / 2]
                        h_avg, s_avg, i_avg = get_forehead_hsv_vectors(curr_frame, roi_frac, diff_x, diff_y)
                        H_mat[count-1][i+3*j]=h_avg
                        S_mat[count-1][i + 3 * j]=s_avg
                        I_mat[count-1][i + 3 * j]=i_avg
                if (count > window_size) & (count % fps == 0):
                    vec_2_calc = H_mat[count - window_size:count]
                    est_hr_10_sec = get_estimated_heart_rate(vec_2_calc, window_size)
                    hr_vec.append(est_hr_10_sec)
                # advance the trackers
                p0 = (p1[st == 1]).reshape(-1, 1, 2)
            prv_frame = curr_frame
            prv_gray = curr_gray
            count += 1
            if count%(fps*2)==0 and count%(fps*10)!=0:
                print ('-'),
            if count%(fps*10)==0:
                print('%d [s]'%(count/fps)),
        else:
            if (count - 1) % fps != 0:
                vec_2_calc = H_mat[count - window_size:count]
                est_hr_10_sec = get_estimated_heart_rate(vec_2_calc, window_size)
                hr_vec.append(est_hr_10_sec)
                print('')
            break
    print('HR detection completed')
    est_hr = get_estimated_heart_rate(H_mat, count)
    print("Estimated heart rate is: %d [bpm]" % est_hr)
    vidcap.release()
    hr_vec_print = create_hr_signal(hr_vec, count)
    print('Working on output video')
    print_hr_to_video(vid_name, outputVid, hr_vec, hr_vec_print, face_roi, forehead_roi, face_roi_orig,count)
    print('Output video done')


def get_points_to_track(first_frame_gray, face_roi):
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    roi_gray = first_frame_gray[face_roi[2]:face_roi[3], face_roi[0]:face_roi[1]]
    # roi = first_frame[150:870, 1170:1670]

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
    # normlize
    for i in range(len(p0)):
        p0[i][0][0] += face_roi[0]
        p0[i][0][1] += face_roi[2]
    return p0


def print_frame_with_trackers(p0, p1, st, counter, frame, color, mask, diffx, diffy, forehead_roi, face_roi):
    for i, (new, old) in enumerate(zip(p1[st == 1], p0[st == 1])):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    frame = cv2.rectangle(frame, (forehead_roi[0] + diffx, forehead_roi[2] + diffy),
                          (forehead_roi[1] + diffx, forehead_roi[3] + diffy), (0, 255, 0))
    frame = cv2.rectangle(frame, (face_roi[0] + diffx, face_roi[2] + diffy),
                          (face_roi[1] + diffx, face_roi[3] + diffy), (255, 0, 0))
    return frame


def get_movement_from_trackers(p0, p1):
    diff_x = []
    diff_y = []
    for i in range(len(p1)):
        diff_x.append(p1[i][0][0] - p0[i][0][0])
        diff_y.append(p1[i][0][1] - p0[i][0][1])
    avg_diff_x = np.around(np.average(diff_x)).astype(np.int)
    avg_diff_y = np.around(np.average(diff_y)).astype(np.int)
    return avg_diff_x, avg_diff_y


def get_forehead_hsv_vectors(frame, roi_forehead, diff_x, diff_y):
    forehead = frame[roi_forehead[2] + diff_y:roi_forehead[3] + diff_y,
               roi_forehead[0] + diff_x:roi_forehead[1] + diff_x]
    forehead_mask = get_bodyColor_mask(forehead, None)
    forehead = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
    h_avg = np.average(forehead[:, :, 0], weights=forehead_mask)
    s_avg = np.average(forehead[:, :, 1], weights=forehead_mask)
    v_avg = np.average(forehead[:, :, 2], weights=forehead_mask)
    return h_avg, s_avg, v_avg


def get_forehead_hsi_vectors(frame, roi_forehead, diff_x, diff_y):
    forehead = frame[roi_forehead[2] + diff_y:roi_forehead[3] + diff_y,
               roi_forehead[0] + diff_x:roi_forehead[1] + diff_x]
    i_avg = np.average(forehead[:, :, :])
    s_avg = 1 - np.average(forehead[:, :, 1])
    h_avg = np.average(forehead[:, :, 0])
    return h_avg, s_avg, i_avg


def get_forehead_rgb_vectors(frame, roi_forehead, diff_x, diff_y):
    forehead = frame[roi_forehead[2] + diff_y:roi_forehead[3] + diff_y,
               roi_forehead[0] + diff_x:roi_forehead[1] + diff_x]
    r_avg = np.average(forehead[:, :, 2])
    g_avg = np.average(forehead[:, :, 1])
    b_avg = np.average(forehead[:, :, 0])
    return r_avg, g_avg, b_avg


def get_estimated_heart_rate(color_space, count):
    lf = np.round(count / 40)
    hf = np.round(count / 15) + 1
    # lf = np.round(count / 40) + 1
    # hf = np.round(count / 10) + 1
    c=[]
    for i in range(len(color_space[0])):
        c.append(np.zeros(len(color_space)))
        for j in range(len(color_space)):
            c[i][j]=color_space[j][i]
    est_heart_rate=[]
    for i in range(len(c)):
        fft_of_channel = abs(fft(c[i]))  # get fft of channel
        fft_of_channel = fft_of_channel[lf:hf]  # cut out  unnecessary frequencies
        max_f = np.where(fft_of_channel == np.amax(fft_of_channel))[0][0] + lf  # find max normalized frequency
        est_heart_rate.append(max_f * (60*fps) / count)
    est_heart_rate=np.median(est_heart_rate)
    # hr_vec_print = np.zeros(abs(fft(color_space)).shape)
    # hr_vec_print[max_f] = np.amax(fft_of_channel)
    # hr_vec_print = ifft(hr_vec_print)
    # hr_vec_print = hr_vec_print / np.amax(hr_vec_print)

    return est_heart_rate


def select_roi(first_frame):
    face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    f = face_cas.detectMultiScale(gray_frame, 1.3, 5)
    (x, y, w, h) = f[0]
    face_roi_orig = [x, x + w, y, y + h]

    y_min = int(y)
    y_max = int(y + h)
    x_min = int(x + 0.17 * w)
    x_max = int(x + 0.71 * w)
    face_roi = [x_min, x_max, y_min, y_max]
    y_min = int(y + 0.05 * h)
    y_max = int(y + 0.2 * h)
    x_min = int(x + 0.22 * w)
    x_max = int(x + 0.68 * w)
    forehead_roi = [x_min, x_max, y_min, y_max]
    return face_roi, forehead_roi, face_roi_orig


def print_hr_to_video(input_vid_name, output_vid_name, hr_print, hr_show, face_roi, forehead_roi, face_roi_orig,count_final):
    vidcap = cv2.VideoCapture(input_vid_name)

    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    font_size = int(min(frame_width, frame_height) / 100)

    out = cv2.VideoWriter(output_vid_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
    color = np.random.randint(0, 255, (100, 3))
    count = 0

    while True:
        success, curr_frame = vidcap.read()
        if success:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            if count == 0:  # Reading the first frame
                # Get points to track
                p0 = get_points_to_track(curr_gray, face_roi)
                first_p0 = p0
                p1 = p0
                st = np.uint8(np.ones(p0.shape[0]))
                prv_gray = curr_gray
                diff_y = 0
                diff_x = 0
            else:
                # Track
                p1, st, err = cv2.calcOpticalFlowPyrLK(prv_gray, curr_gray, p0, None, **lk_params)
                prv_gray = curr_gray
                # Find the difference in the roi
                diff_x, diff_y = get_movement_from_trackers(first_p0, p1)
            hr_box_len = min(int(curr_frame.shape[0] * 0.2), int(curr_frame.shape[1] * 0.2))
            curr_frame = cv2.rectangle(curr_frame, (0, 0), (hr_box_len, hr_box_len), (255, 255, 255), cv2.FILLED)
            curr_frame = cv2.putText(curr_frame, str(int(hr_print[max((count - window_size) / 30, 0)])),
                                     (int(hr_box_len * 0.05), int(hr_box_len * 0.8)), cv2.FONT_HERSHEY_PLAIN, font_size,
                                     (0, 0, 255), thickness=5)

            mask = get_bodyColor_mask(curr_frame, face_roi_orig)
            if count <= hr_show.shape[0] - 1:
                mask = np.uint8(mask * (np.real(hr_show[count]) * 15))
            else:
                mask = np.uint8(mask * (np.real(hr_show[hr_show.shape[0] - 1]) * 15))
            curr_frame[:, :, 0] += mask

            print_frame_with_trackers(p0, p1, st, count, curr_frame, color, mask, diff_x, diff_y, forehead_roi,
                                      face_roi_orig)

            if count != 0:
                # advance the trackers
                p0 = (p1[st == 1]).reshape(-1, 1, 2)

            out.write(curr_frame)
            count += 1
            if count%60==0:
                print('%d %s' % ((int(count * 100 / count_final)),'%'))
        else:
            print('100%')
            break
    vidcap.release()
    out.release()


def get_bodyColor_mask(frame, face_roi):
    lc = np.array([0, 30, 100])
    hc = np.array([20, 200, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lc, hc)

    if face_roi != None:
        mask2 = np.zeros(mask.shape)
        deltaX = face_roi[1] - face_roi[0]
        mask2[:, face_roi[0] - deltaX * 2:face_roi[1] + deltaX * 2] = 255
        mask = cv2.bitwise_and(mask, mask, mask=np.uint8(mask2))
    mask = mask / 255
    return mask


def create_hr_signal(hr_vec, count):
    hr_vec = np.multiply(hr_vec, window_size) / (fps*60)
    phase = np.zeros(count)
    res = np.zeros(count)
    help_idx = 0
    for i in range(count):
        if i != 0:
            phase_v = np.float32(hr_vec[help_idx]) / window_size
            phase[i] = phase[i - 1] + phase_v * 2 * np.pi
            res[i] = np.cos(phase[i])
            if i > window_size and i % fps == 0:
                help_idx += 1
    return res


def handle_arg(args):
    if args.WinLen!=None:
        window_size=args.WinLen
    if args.fps != None:
        fps = args.fps
    x=1
    y=1
    if args.xParts!=None:
        x=args.xParts
    if args.yParts!=None:
        y=args.yParts
    return args.inputVid, args.outputVid, x,y

if __name__ == "__main__":
    main()
