import random
import numpy as np
import torch
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
from .general import xyxy2xywh


class TrackerState():
    INIT = 0
    WAIT = 1
    HIT  = 2

class Tracker(object):
    global_id = 0
    pending_timeout = 30
    def __init__(self, xyxy):
        self.xyxy = xyxy
        # print("new Tracker : ", self.xyxy)
        self.x, self.y, self.w, self.h = self.det2xywh(self.xyxy)
        self.left_top = (self.x, self.y)
        self.bottom_mid = (self.x + self.w / 2, self.y + self.h)
        self.pending_time = 0
        self.bottom_mid_history = []
        self.trakcer_id = Tracker.global_id
        self.state = TrackerState.HIT
        self.refresh_flag = True
        self.color = tuple(np.random.choice(range(256), size=3))
        Tracker.global_id += 1

    def pending(self):
        self.pending_time += 1
        self.state = TrackerState.WAIT
        self.refresh_flag = True


    def update_point(self, xyxy):
        self.pending_time = 0
        self.refresh_flag = True
        self.state = TrackerState.HIT

        self.xyxy = xyxy
        self.x, self.y, self.w, self.h = self.det2xywh(self.xyxy)
        self.left_top = (self.x, self.y)
        self.bottom_mid = (self.x + self.w / 2, self.y + self.h)
        self.bottom_mid_history.append(self.bottom_mid)

        if len(self.bottom_mid_history) > 10:
            self.bottom_mid_history.pop(0)
        
    
    def get_iou(self, xyxy2):
        boxA = [0,0,0,0]
        boxB = [0,0,0,0]
        
        # print(f"{xyxy2[0] = }")
        boxB[0] = int(xyxy2[0])
        boxB[1] = int(xyxy2[1])
        boxB[2] = int(xyxy2[2])
        boxB[3] = int(xyxy2[3])
        boxA[0] = int(self.xyxy[0])
        boxA[1] = int(self.xyxy[1])
        boxA[2] = int(self.xyxy[2])
        boxA[3] = int(self.xyxy[3])

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def det2xywh(self, xyxy):
        x = int(xyxy[0])
        y = int(xyxy[1])
        w = int(xyxy[2]) - x
        h = int(xyxy[3]) - y
        return x, y, w, h

class MamaTracker(object):
    def __init__(self):
        self.tracker_list = []

        self.roi = [(0,0), (1,0), (1,1), (0,1)]
        self.iou_thres = 0.1

    def is_in_roi(self, xyxy):
        # to tell the point is in roi or not
        return True

    def matching(self, dets):
        for tracker in self.tracker_list:
            tracker.refresh_flag = False
        
        for det in reversed(dets):
            *xyxy, conf, cls_ = det
            if self.is_in_roi(xyxy):     
                max_iou = -1
                best_tracker = None
                for tracker in self.tracker_list:
                    # print(f"\nmapping tracker : \n->{tracker.trakcer_id = }\n->{tracker.xyxy}\n->{xyxy}")
                    if (tracker.refresh_flag == True):
                        continue
                    iou = tracker.get_iou(xyxy)
                    if iou > self.iou_thres and iou > max_iou:
                        max_iou = iou
                        best_tracker = tracker
                if best_tracker is None:
                    self.tracker_list.append(Tracker(xyxy))
                else:
                    best_tracker.update_point(xyxy)

        del_list = []
        for cnt, tracker in enumerate(self.tracker_list):
            if tracker.refresh_flag:
                pass
            else : 
                tracker.pending()
                if tracker.pending_time >= Tracker.pending_timeout:
                    del_list.append(cnt)

        for del_num in reversed(del_list):
            del(self.tracker_list[del_num])

    @property
    def tracker_boxes(self):
        detections = []
        for tracker in self.tracker_list:
            detections.append(tracker.xyxy)
        return detections
