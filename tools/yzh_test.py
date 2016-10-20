import numpy as np

def boxs_merge_yaho(boxes, scores, overlapThresh=0.2):
    # from the YAHO model but not good
    scores = np.array(scores)
    if len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    result_boxes = []
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    result_score = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        # overlap = (w * h) / (area[idxs[:last]]  - w * h + area_array)

        overlap = (w * h) / (area[idxs[:last]])
        delete_idxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        result_score.append(scores[delete_idxs].mean())
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        ave_prob = 0
        width = x2[i] - x1[i] + 1
        height = y2[i] - y1[i] + 1
        for idx in delete_idxs:
            ave_prob += boxes[idxs[idx]][4]
            if (boxes[idxs[idx]][0] < xmin):
                xmin = boxes[idxs[idx]][0]
            if (boxes[idxs[idx]][1] < ymin):
                ymin = boxes[idxs[idx]][1]
            if (boxes[idxs[idx]][2] > xmax):
                xmax = boxes[idxs[idx]][2]
            if (boxes[idxs[idx]][3] > ymax):
                ymax = boxes[idxs[idx]][3]
        if (x1[i] - xmin > 0.1 * width):
            xmin = x1[i] - 0.1 * width
        if (y1[i] - ymin > 0.1 * height):
            ymin = y1[i] - 0.1 * height
        if (xmax - x2[i] > 0.1 * width):
            xmax = x2[i] + 0.1 * width
        if (ymax - y2[i] > 0.1 * height):
            ymax = y2[i] + 0.1 * height
        result_boxes.append([xmin, ymin, xmax, ymax, ave_prob / len(delete_idxs)])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, delete_idxs)
        scores = np.delete(scores, delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return result_boxes, result_score


def boxs_merge(boxes, scores, iou=.3):
    # written by yzh for box merge using iou
    rst_sco = []
    rst_boxes = []
    scores = np.array(scores)
    boxes = np.array(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    while len(scores):
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        i = scores.argmax()
        left = [id for id in range(len(scores)) if id != i]
        xx1 = np.maximum(x1[i], x1[left])
        yy1 = np.maximum(y1[i], y1[left])
        xx2 = np.minimum(x2[i], x2[left])
        yy2 = np.minimum(y2[i], y2[left])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(boxes) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        overlaps = (w * h) / (areas[left])
        cor=[boxes[i,:2],boxes[i,2:]]
        rm=[]
        for id,lap in enumerate(overlaps):
            if lap>iou:
                if id>=i:id+=1
                cor+=[boxes[id,:2],boxes[id,2:]]
                rm.append(id)
        if rm==[]:
            rm.append(i)
        rst_sco.append(scores[rm])
        rst_boxes.append(boxes[scores[rm].argmax(0)])

        x1 = np.delete(x1, rm)
        x2 = np.delete(x2, rm)
        y1 = np.delete(y1, rm)
        scores = np.delete(scores, rm,0)
        y2 = np.delete(y2, rm)
        boxes = np.delete(boxes, rm,0)
        areas = np.delete(areas, rm,0)


        cor=np.array(cor)
        #rst_boxes.append([cor[:,0].min(),cor[:,1].min(),cor[:,0].max(),cor[:,0].min()])
    for id,score in enumerate(rst_sco):
        rst_sco[id]=np.mean(score)
    return rst_boxes,rst_sco

boxs_merge([[1,2,3,4],[1.5,2.5,3.5,4.5],[2,3,4,5]],[[1],[2],[3]])