import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPoint

def py_cpu_nms(boxlist, thresh):
    coords = boxlist.bbox  # [[],[],,,[]],shape:193*8
    scores = boxlist.get_field("scores")  # shape:193, [0.7,0.35,,,0.85]
    labels = boxlist.get_field("labels")  # shape:193,[1,1,1,,,1]
    coords = np.array(coords.cpu())
    scores = np.array(scores.cpu())

    x1 = coords[:,0]
    y1 = coords[:,1]
    x2 = coords[:,2]
    y2 = coords[:,3]
    x3 = coords[:,4]
    y3 = coords[:,5]
    x4 = coords[:,6]
    y4 = coords[:,7]

    areas = abs((x3-x1)*(y3-y1))
    keep = []
    index = scores.argsort()[::-1]

    while index.size >0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x33 = np.minimum(x3[i], x3[index[1:]])
        y33 = np.minimum(y3[i], y3[index[1:]])

        w = np.maximum(0, x33-x11+1)    # the weights of overlap
        h = np.maximum(0, y33-y11+1)

        overlaps = w*h
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
        idx = np.where(ious<=thresh)[0]  # 0表示取得是索引而不是值
        index = index[idx+1]
    return boxlist[keep]

def cal_iou(box1, box2):
    a = box1.reshape(4, 2)
    b = box2.reshape(4, 2)
    union_poly = np.concatenate((a,b))
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull
    if not poly1.intersects(poly2): #如果两四边形不相交
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = MultiPoint(union_poly).convex_hull.area
        iou=float(inter_area) / union_area
    return iou

def py_cpu_nms_polygon(boxlist, thresh):
    """
    如果按照polygon.intersects的方式计算，那么需要写for循环计算index[i]与每个index[1:]的交集，时间耗费较久
    """
    coords = boxlist.bbox
    scores = boxlist.get_field("scores")
    # print(scores)
    coords = np.array(coords.cpu()).reshape(-1,4,2)
    scores = np.array(scores.cpu())

    area = []
    for coord in coords:
        coord = Polygon(coord)
        area.append(coord)
    area = np.array(area)

    keep = []
    index = scores.argsort()[::-1]
    while index.size >0:
        i = index[0]
        keep.append(i)
        iou_list = []
        for idx in index[1:]:
            iou = cal_iou(coords[i,:,:],coords[idx,:,:])
            iou_list.append(iou)
        iou_array = np.array(iou_list)
        idx = np.where(iou_array<=thresh)[0]
        index = index[idx+1]
    return boxlist[keep]

    # line1=[2,0,2,2,0,0,0,2]  #四边形四个点坐标的一维数组表示，[x,y,x,y....]
    # a=np.array(line1).reshape(4, 2)  #四边形二维坐标表示
    # poly1 = Polygon(a).convex_hull #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    # print(Polygon(a).convex_hull) #可以打印看看是不是这样子
    
    # line2=[1,1,4,1,4,4,1,4]
    # b=np.array(line2).reshape(4, 2)
    # poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)
    
    # union_poly = np.concatenate((a,b))  #合并两个box坐标，变为8*2
    # #print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)   #包含两四边形最小的多边形点
    # if not poly1.intersects(poly2): #如果两四边形不相交
    #     iou = 0
    # else:
    #     try:
    #         inter_area = poly1.intersection(poly2).area  #相交面积
    #         print(inter_area)
    #         #union_area = poly1.area + poly2.area - inter_area
    #         union_area = MultiPoint(union_poly).convex_hull.area
    #         print(union_area)
    #         if union_area == 0:
    #             iou= 0
    #         #iou = float(inter_area) / (union_area-inter_area) #错了
    #         iou=float(inter_area) / union_area
    #         # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
    #         # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积 
    #         # 第二种： 交集 / 并集（常见矩形框IOU计算方式） 
    #     except shapely.geos.TopologicalError:
    #         print('shapely.geos.TopologicalError occured, iou set to 0')
    #         iou = 0
