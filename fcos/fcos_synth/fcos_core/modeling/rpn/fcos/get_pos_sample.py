import torch

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    # poi为anchor点，s_poi为线段的第一个点坐标，e_poi为第二个线段的点坐标，且线段是顺时针顺序
    if s_poi[1]==e_poi[1]: # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[0]==e_poi[0]:  # 竖直线段，只有anchor点在线段左边且y位于线段y之间
        if ((s_poi[1]-poi[1]) * (e_poi[1]-poi[1]) < 0) and (poi[0] < s_poi[0]):
            return True
        else:
            return False
    if s_poi[1]>=poi[1] and e_poi[1]>=poi[1]:  # anchor点在线段上方
        return False
    if s_poi[1]<=poi[1] and e_poi[1]<=poi[1]:  # anchor点在线段下方
        return False
    if s_poi[0]<poi[0] and e_poi[0]<poi[0]:  # anchor点在线段的右侧
        return False
    return True

def isPoiWithinPoly(gts,polys_x,polys_y):  # 一个anchor，很多polygon
    # 输入：点，多边形三维数组
    # pois=torch.size[3,8] polys_x/polys_y:torch.size[21486]
    result = []
    gts = gts.reshape(-1,4,2)
    for i in range(len(polys_x)):
        poi = [polys_x[i],polys_y[i]]
        temp_list = []
        for gt in gts:
            sinsc = 0
            for i in range(len(gt)-1):
                s_poi=gt[i]
                e_poi=gt[i+1]
                if isRayIntersectsSegment(poi,s_poi,e_poi):
                    sinsc += 1
            if isRayIntersectsSegment(poi,gt[len(gt)-1],gt[0]):
                sinsc += 1
            temp_list.append(True if sinsc % 2 == 1 else False)
        result.append(temp_list)
    result = torch.tensor(result)  # [21486,3]
    return result

    sinsc=0  # 交点个数
    for i in range(len(poly)-1):  # [0,len-1] 顺时针数组
        s_poi=poly[i]
        e_poi=poly[i+1]
        if isRayIntersectsSegment(poi,s_poi,e_poi):
            sinsc += 1 #有交点就加1

    return True if sinsc % 2 == 1 else  False