from styx_msgs.msg import TrafficLight
import rospy
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

path = 'ssd_graph.pb'

def findNoneZero(rgb_image):
    rows,cols,_ = rgb_image.shape
    counter = 0
    for row in range(rows):
        for col in range(cols):
            pixels = rgb_image[row,col]
            if sum(pixels)!=0:
                counter = counter+1
    return counter

def red_green_yellow(rgb_image):
    hsv = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:,:,1])
    area = 32*32
    avg_saturation = sum_saturation / area 
    
    sat_low = int(avg_saturation*1.3)
    val_low = 140
    # Green
    lower_green = np.array([36, sat_low, val_low])
    upper_green = np.array([71, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_result = cv2.bitwise_and(rgb_image, rgb_image, mask = green_mask)
    # Yellow
    lower_yellow = np.array([18, sat_low, val_low])
    upper_yellow = np.array([36, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask=yellow_mask)
    # Red 
    lower_red = np.array([0, sat_low, val_low])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask = red_mask)
    
    sum_green = findNoneZero(green_result)
    sum_red = findNoneZero(red_result)
    sum_yellow = findNoneZero(yellow_result)
    if sum_red >= sum_yellow and sum_red>=sum_green:
        return 1
    if sum_yellow>=sum_green:
        return 2
    return 3

def major(lt):
    cand=-1
    times=0
    for i in range(len(lt)):
        if times==0:
            cand = lt[i]
            times = 1
        elif lt[i]==cand:
            times+=1
        else:
            times-=1
    return cand

def detect_color(image, boxes):
    a = []
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        cropImg = image[int(bot):int(top), int(left):int(right), :]
        standard_im = cv2.resize(cropImg,(32,32))
        temp = red_green_yellow(standard_im)
        a.append(temp)
    #for j in range(len(a)):
        #rospy.logwarn('nihao0 = %d', a[j])
    b = major(a)
    if b==1:
        return TrafficLight.RED
    elif b==2:
        return TrafficLight.YELLOW
    else:
        return TrafficLight.GREEN

def get_graph(file_path):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as f:
            g = f.read()
            graph_def.ParseFromString(g)
            tf.import_graph_def(graph_def, name='')
    return graph

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.my_graph = get_graph(path)
        self.image_tensor = self.my_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.my_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.my_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.my_graph.get_tensor_by_name('detection_classes:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        color = TrafficLight.UNKNOWN
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_ex = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.my_graph) as sess:
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor : image_ex})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            n = len(classes)
            arrs = []
            flag = True
            for i in range(n):
                if scores[i] >= 0.2 and classes[i] == 10:
                    arrs.append(i)
            if len(arrs)==0:
                flag = False
            de_boxes = boxes[arrs, ...]
            if flag:
                w, h = Image.fromarray(image).size
                boxes_copy = np.zeros_like(de_boxes)
                boxes_copy[:, 0] = de_boxes[:, 0] * h
                boxes_copy[:, 1] = de_boxes[:, 1] * w
                boxes_copy[:, 2] = de_boxes[:, 2] * h
                boxes_copy[:, 3] = de_boxes[:, 3] * w
                color = detect_color(image, boxes_copy)
        return color
