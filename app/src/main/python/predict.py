import os

import numpy as np
import base64
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
from scipy import ndimage
import copy, re
import math
from os.path import dirname, join
class Latex(object):
    def __init__(self, model_dir=None, mean_train=None, std_train=None, plotting=False, verbose=False):
        # tf.logging.set_verbosity(tf.logging.WARN)

        self.model_dir = model_dir
        self.mean_train = mean_train
        self.std_train = std_train
        self.plotting = plotting
        self.verbose = verbose
        self.label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', 'leq', 'neq', 'geq',
                            'alpha',
                            'beta', 'lambda', 'lt', 'gt', 'x', 'y']
        self.ltokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', '#leq', '#neq', '#geq',
                        '#alpha',
                        '#beta', '#lambda', '#lt', '#gt', 'x', 'y', '^', '#frac', '{', '}', ' ']
        self.nof_labels = len(self.label_names)
        self.labels_dict = dict()
        i = 0
        for label in self.label_names:
            self.labels_dict[label] = i
            i += 1
        self.classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn, model_dir=self.model_dir
        )

        self.seq_sess = tf.Session()



    def train(self, train_images, train_labels, steps):

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_images},
            y=train_labels,
            batch_size=500,
            num_epochs=None,
            shuffle=True
        )
        self.classifier.train(
            input_fn=train_input_fn,
            steps=steps,
        )

    def normalize_single(self, symbol):
        symbol = np.copy(symbol).astype(np.float32)

        # range 0-1
        symbol /= np.max(symbol)

        rows, cols = symbol.shape
        # scale to 40x40
        inner_size = 40
        if rows > cols:
            factor = inner_size / rows
            rows = inner_size
            cols = int(round(cols * factor))
            cols = cols if cols > 2 else 2
            inner = cv2.resize(symbol, (cols, rows))
        else:
            factor = inner_size / cols
            cols = inner_size
            rows = int(round(rows * factor))
            rows = rows if rows > 2 else 2
            inner = cv2.resize(symbol, (cols, rows))

        # pad to 48x48
        outer_size = 48
        colsPadding = (int(math.ceil((outer_size - cols) / 2.0)), int(math.floor((outer_size - cols) / 2.0)))
        rowsPadding = (int(math.ceil((outer_size - rows) / 2.0)), int(math.floor((outer_size - rows) / 2.0)))
        outer = np.pad(inner, (rowsPadding, colsPadding), 'constant', constant_values=(1, 1))

        # center the mass
        shiftx, shifty = self.getBestShift(outer)
        shifted = self.shift(outer, shiftx, shifty)
        return shifted

    def getBestShift(self, img):
        inv = (img)
        cy, cx = ndimage.measurements.center_of_mass(inv)

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows), borderValue=1)
        return shifted

    def add_rectangles(self, img, bounding_boxes1):
        img_color = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
        for bounding_box in bounding_boxes1:
            xmin = bounding_box[0]
            xmax = bounding_box[0] + bounding_box[2]
            ymin, ymax = bounding_box[1], bounding_box[1] + bounding_box[3]
            img_color[ymin - 12:ymin + (bounding_box[3] + 12), xmin - 12:xmin + (bounding_box[2] + 12)] = [255, 0, 0]

        return img_color

    def add_rectangles1(self, img, bounding_boxes1):
        img_color = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
        for bounding_box in bounding_boxes1:
            xmin = bounding_box['xmin']
            xmax = bounding_box['xmax']
            ymin, ymax = bounding_box['ymin'], bounding_box['ymax']
            img_color[ymin, xmin:xmax] = [255, 0, 0]
            img_color[ymax - 1, xmin:xmax] = [255, 0, 0]
            img_color[ymin:ymax, xmin] = [255, 0, 0]
            img_color[ymin:ymax, xmax - 1] = [255, 0, 0]
        return img_color


    def cnn_model_fn(self, features, labels, mode):
        input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[7, 7],
            padding="same",
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2
        )

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[7, 7],
            padding="same",
            activation=tf.nn.relu
        )

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2
        )

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[7, 7],
            padding="same",
            activation=tf.nn.relu
        )

        pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=[2, 2],
            strides=2
        )

        pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 128])
        dense = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        logits = tf.layers.dense(inputs=dropout, units=self.nof_labels)

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.nof_labels)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"]
            )
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def get_bounding_boxes(self):
        ret, thresh = cv2.threshold(self.formula, 70, 255, cv2.THRESH_BINARY_INV)


        ctrs, ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w = int(48)
        h = int(48)
        train_data = []
        # print(len(cnt))
        rects = []
        i = 0
        for c in cnt:

            if cv2.contourArea(c) > 50:

                i = i + 1
                x, y, w, h = cv2.boundingRect(c)
                rect = [x, y, w, h]
                rects.append(rect)
        # print(rects)
        bool_rect = []
        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (
                            rec[1] + rec[3] + 10) and \
                            rec[1] < (r[1] + r[3] + 10):
                        flag = 1
                    l.append(flag)
                if rec == r:
                    l.append(0)
            bool_rect.append(l)
        # print(bool_rect)
        dump_rect = []
        for i in range(0, i):
            for j in range(0, i):
                if bool_rect[i][j] == 1:
                    area1 = rects[i][2] * rects[i][3]
                    area2 = rects[j][2] * rects[j][3]
                    if (area1 == min(area1, area2)):
                        dump_rect.append(rects[i])
        # print(len(dump_rect))
        final_rect = [i for i in rects if i not in dump_rect]

        formula_rects = self.add_rectangles(self.formula, final_rect)


        self.bounding_boxes = final_rect

    def post_process_latex(self, formula_text):
        formula_text = formula_text.replace("=", " = ")
        for symbol in ["leq", "neq", "geq"]:
            formula_text = formula_text.replace(symbol, " \\" + symbol + " ")
        for symbol in ["lambda", "alpha", "beta"]:
            formula_text = formula_text.replace(symbol, "\\" + symbol)
        formula_text = formula_text.replace("#lt", "<")
        formula_text = formula_text.replace("#gt", ">")
        return formula_text


    def normalize(self):
        self.possible_symbol_img = []
        self.pred_pos = []
        for bounding_box in self.bounding_boxes:
            xmin, xmax = bounding_box[0], bounding_box[0] + bounding_box[2]
            ymin, ymax = bounding_box[1], bounding_box[1] + bounding_box[3]
            dy = ymin + bounding_box[3]
            dx = xmin + bounding_box[2]

            normalized = self.normalize_single(self.formula[ymin:ymax, xmin:xmax])
            normalized -= self.mean_train
            normalized /= self.std_train

            self.possible_symbol_img.append(normalized)
            self.pred_pos.append(bounding_box)

    def predict(self, formula):
        self.formula = formula
        self.get_bounding_boxes()
        self.normalize()

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(self.possible_symbol_img)},
            shuffle=False
        )

        pred_results = self.classifier.predict(input_fn=eval_input_fn)
        good_bounding_boxes = []
        formula_text = ""

        pred_pos = self.pred_pos

        skip = []
        c = 0

        lastYmin = None
        lastYmax = None
        for pred_result, pos in zip(pred_results, pred_pos):
            symbol_no = pred_result['classes']
            symbol = self.label_names[symbol_no]
            acc = pred_result['probabilities'][symbol_no]
            if self.verbose:
                print("Recognized a %s with %.2f %% accuracy" % (symbol, acc * 100))
            formula_text += symbol



        return {'formula': self.post_process_latex(formula_text)}

def predict(bmp):

    d = base64.b64decode(bmp)

    np_data = np.fromstring(d,np.uint8)

    image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)




    mean_train = np.load(join(dirname(__file__), "train_images_mean.npy"))
    std_train = np.load(join(dirname(__file__), "train_images_std.npy"))

    model = Latex("model", mean_train, std_train)





    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,201,12)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    dilate = cv2.dilate(thresh, kernel, iterations=2)


    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x,y,w,h = cv2.boundingRect(c)
        area = w * h
        ar = w / float(h)
        if area > 5000 and area < 500000 and ar < 6:
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)

    # Bitwise-and input image and mask to get result
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask==0] = (255,255,255) # Color background white
    d=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(d,120,255,cv2.THRESH_BINARY)
    imS = cv2.resize(thresh1, (960, 540))


    formula = thresh1

    latex = model.predict(formula)

    return latex['formula']



