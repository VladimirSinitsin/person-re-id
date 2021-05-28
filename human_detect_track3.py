"""
Version3
tracking + reidentification.
Tracking is done using last k=15 frames.

Change from human_detect_track3: some changes in find()
Used matchTemplate() to check the result of tracking.
Sometimes, 2 people very close to each other are tracked as
the same person, so this makes sure that this error does not occur.
"""
import numpy as np
import tensorflow as tf
import cv2
import time
import shutil
import os

from pathlib import Path
from PIL import ImageFont, ImageDraw, Image\

from run import Reid
# from run import main2
from importlib import import_module


SCRIPT_PATH = str(Path(__file__).parent.absolute())


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.reid = Reid()
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def process_frame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        im_height, im_width, _ = image.shape
        boxes_list = [None for _ in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (
            int(boxes[0, i, 0] * im_height), int(boxes[0, i, 1] * im_width), int(boxes[0, i, 2] * im_height),
            int(boxes[0, i, 3] * im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

    def find(self, img, boxes_cur, boxes_prev, box):
        # print('## Find called')
        cv2.imwrite('./temporaryImg.jpg', img)

        past_ppl = './past_ppl'
        folders = os.listdir(past_ppl)

        for folder in folders:
            files = os.listdir(past_ppl + '/' + folder)
            same = 0
            diff = 0
            num_of_files = len(files)
            for f in range(num_of_files):
                if f % 10 != 0:
                    continue
                ret = self.reid.compare('./temporaryImg.jpg', './past_ppl/' + folder + '/' + str(f + 1) + '.jpg')

                if ret:
                    same += 1
                else:
                    diff += 1

            p = 100 * float(same) / float(same + diff)
            if p > 70:
                person_no = len(files) + 1
                cv2.imwrite(past_ppl + '/' + folder + '/' + str(person_no) + '.jpg', img)
                boxes_cur[int(folder)][0] = box
                boxes_prev[int(folder)] = -1
                return folder

        person_no = len(folders)
        os.makedirs(past_ppl + '/' + str(person_no))
        cv2.imwrite(past_ppl + '/' + str(person_no) + '/1.jpg', img)
        boxes_cur.append([box])

        return person_no


def make_empty_folder(out_path):
    print(out_path)
    # make path if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)  # empty it if anything there
        return
    shutil.rmtree(out_path)
    make_empty_folder(out_path)


def iou(box1, box2):
    xa = max(box1[1], box2[1])
    ya = max(box1[0], box2[0])
    xb = min(box1[3], box2[3])
    yb = min(box1[2], box2[2])

    inter_area = max(0, xb - xa) * max(0, yb - ya)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = float(inter_area) / float(box1_area + box2_area - inter_area)

    # return the intersection over union value
    return iou


def draw_text(img, box, num):
    text_to_draw = f'человек {num}'
    font_w, font_h = FONT.getsize(text_to_draw)
    y_min, x_min, y_max, x_max = box
    xy_coords = (x_min, y_min - font_h - 2)

    cv2.rectangle(img, (x_min - 1, y_min - 1), (x_max + 1, y_max + 1), (0, 255, 0), 2)
    cv2.rectangle(img, (x_min - 1, y_min - font_h - 1), (x_min + font_w + 1, y_min), (0, 255, 0), -1)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((xy_coords[0] + 1, xy_coords[1] + 1), text_to_draw, font=FONT, fill=(0, 0, 0))
    draw.text(xy_coords, text_to_draw, font=FONT, fill=(255, 255, 255))
    img = np.array(img_pil)

    # cv2.imwrite(f'{SCRIPT_PATH}/{records}/{count}.jpg', img)
    # count += 1

    return img


#   model_path = '/path/to/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb'
FONT = ImageFont.truetype('./Fallout2Cyr.ttf', 16)
model_path = './model/frozen_inference_graph.pb'

past_ppl = '/past_ppl'
records = '/records'

make_empty_folder(SCRIPT_PATH + past_ppl)
make_empty_folder(SCRIPT_PATH + records)

odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.8
iou_threshold = 0.6
cap = cv2.VideoCapture('./video2.avi')

# maximum number of previous frames to check iou with
k = 15

# this will store the bounding boxes detected in the previous frame.
boxes_prev = []
framenum = 1
start_time = time.time()  # seconds
time240 = 0

count = 0

# iterate over frames
while True:
    _, source_img = cap.read()
    img = cv2.resize(source_img, (1280, 720))

    boxes, scores, classes, num = odapi.process_frame(img)
    boxes_cur = []
    for box in boxes_prev:
        boxes_cur.append([-1] + box[:k - 1])

    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box

            # draw the bounding box on the image
            # cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

            cropped_img = img[y_min:y_max, x_min:x_max]

            maxthreshold = -1
            maxindex = 101  # the index in boxes_prev indicating the matching person from the previous k frames.
            maxframe = -1

            for j in range(len(boxes_prev)):
                # Every boxes_prev[j] denotes a person. It is a list of the last k positions of the person j.

                # This previous person has already been alloted to another person in the current frame
                if boxes_prev[j] == -1:
                    continue

                for kk in range(len(boxes_prev[j])):
                    if boxes_prev[j][kk] == -1:  # person was not detected in frame kk
                        continue
                    curr_iou = iou(boxes_prev[j][kk], box)
                    if curr_iou > maxthreshold and curr_iou > iou_threshold:
                        maxthreshold = curr_iou
                        maxindex = j
                        maxframe = kk

            if maxthreshold != -1:
                # Was the tracking correct?
                b = boxes_prev[maxindex][maxframe]
                old_img = img[b[0]:b[2], b[1]:b[3]]
                cur_img = cropped_img

                h, w, d = cur_img.shape
                old_img = cv2.resize(old_img, (w, h))

                res = cv2.matchTemplate(old_img, cur_img, cv2.TM_CCOEFF_NORMED)
                if res[0][0] < 0.45:
                    # Tracking was incorrect
                    maxthreshold = -1

            # maxthreshold != -1 at this point means this person is the same as prevbox in the last frame.
            if maxthreshold != -1:
                # print('tracked ###########')
                boxes_cur[maxindex][0] = box
                boxes_prev[maxindex] = -1

                # also add this image of the person to his previous images
                person_no = len(os.listdir(SCRIPT_PATH + past_ppl + '/' + str(maxindex))) + 1
                cv2.imwrite(SCRIPT_PATH + past_ppl + '/' + str(maxindex) + '/' + str(person_no) + '.jpg', cropped_img)
                img = draw_text(img, box, maxindex)
            else:
                # The person was not present in the previous frame. Add him to a new folder.
                # The folder name should be equal to the index of the person in box_cur.
                person_no = odapi.find(cropped_img, boxes_cur, boxes_prev, box)
                img = draw_text(img, box, person_no)

    num_ppl = len(os.listdir(SCRIPT_PATH + past_ppl))
    # print('#People:   ' + str(num_ppl))
    print('Time for ' + str(framenum) + ' frames: (seconds)')
    print(time.time() - start_time)
    print('\n')

    framenum += 1
    boxes_prev = boxes_cur

    if framenum == 240:
        time240 = time.time() - start_time

    if framenum > 240:
        print('Time for 240 frame:' + str(time240))
        print('\n')

    cv2.imshow("preview", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
