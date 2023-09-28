# def generalized_IOU_loss(y_true, y_predict):
#     #print(tf.keras.backend.eval(y_true))
#     #print(tf.keras.backend.eval(y_true[:, 0]))
#     #(x1p, y1p, x2p, y2p) = y_predict
#     print(tf.keras.backend.eval(y_true))
#     print(tf.keras.backend.eval(y_predict))
#     x1p = tf.keras.backend.eval(y_predict[:, 0])
#     y1p = tf.keras.backend.eval(y_predict[:, 1])
#     x2p = tf.keras.backend.eval(y_predict[:, 2])
#     y2p = tf.keras.backend.eval(y_predict[:, 3])
#     #(x1g, y1g, x2g, y2g) = y_true
#     x1g = tf.keras.backend.eval(y_true[:, 0])
#     y1g = tf.keras.backend.eval(y_true[:, 1])
#     x2g = tf.keras.backend.eval(y_true[:, 2])
#     y2g = tf.keras.backend.eval(y_true[:, 3])
#     #if x2p > x1p and y2p > y1p:
#     x1phat = np.minimum(x1p, x2p)
#     x2phat = np.maximum(x1p, x2p)
#     y1phat = np.minimum(y1p, y2p)
#     y2phat = np.maximum(y1p, y2p)
#     Ag = np.multiply((x2g - x1g), (y2g - y1g))
#     Ap = np.multiply((x2phat - x1phat), (y2phat - y1phat))
#     x1I = np.maximum(x1phat, x1g)
#     x2I = np.minimum(x2phat, x2g)
#     y1I = np.maximum(y1phat, y1g)
#     y2I = np.minimum(y2phat, y2g)
#     I = np.where(np.logical_and(x2I > x1I, y2I > y1I), (x2I - x1I) * (y2I - y1I), 0)
#     #(x2I - x1I) * (y2I - y1I) if (x2I > x1I and y2I > y1I) else 0
#     x1c = np.minimum(x1phat, x1g)
#     x2c = np.maximum(x2phat, x2g)
#     y1c = np.minimum(y1phat, y1g)
#     y2c = np.maximum(y2phat, y2g)
#     Ac = np.multiply((x2c - x1c), (y2c - y1c))
#     U = Ap + Ag - I
#     IoU = np.divide(I, U)
#     GIoU = np.divide(IoU - (Ac - U), Ac)
#     GIOU_loss = 1-GIoU
#     return tf.math.reduce_sum(GIOU_loss)

# def GIoU(bboxes_1, bboxes_2):
#     # https://github.com/shjo-april/Tensorflow_GIoU/blob/master/README.md
#     # https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/DNN_IOU_SEGMENTATION.pdf
#     # 1. calulate intersection over union
#     area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])
#     area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])
    
#     intersection_wh = tf.minimum(bboxes_1[:, 2:], bboxes_2[:, 2:]) - tf.maximum(bboxes_1[:, :2], bboxes_2[:, :2])
#     intersection_wh = tf.maximum(intersection_wh, 0)
    
#     intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
#     union = (area_1 + area_2) - intersection
    
#     ious = intersection / tf.maximum(union, 1e-10)

#     # 2. (C - (A U B))/C
#     C_wh = tf.maximum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.minimum(bboxes_1[..., :2], bboxes_2[..., :2])
#     C_wh = tf.maximum(C_wh, 0.0)
#     C = C_wh[..., 0] * C_wh[..., 1]
    
#     giou = ious - (C - union) / tf.maximum(C, 1e-10)
#     return 1-giou

currMin = 10**6
for a in range(10**6):
    for b in range(10**6//1930):
        for c in range(10**6//2023):
            if a+1930*b+2023*c == 10**6:
                num = a+b+c
                if num < currMin:
                    currMin = num
print(currMin)