# -*- coding: utf-8 -*-
# @Time    : 21-8-15 :05



import io
import os
import cv2
import json
import random
import contextlib
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# for training
# from data.data_augment import random_perspective, box_candidates, TrainTransform

# for vis
from data_augment import random_perspective, box_candidates, TrainTransform

# COCO FORMAT: XYWH
# PASCAL_VOC FORMAT: XYXY

class COCODataset(torch.utils.data.Dataset):

    def __init__(self, cfg, img_size=(640, 640), name="train2017", json_file="instances_train2017.json", preproc=None,
                 no_aug=True, tracking=False, logger=None):

        # super(COCODataset, self).__init__()
        super().__init__()
        self.opt = cfg # configurations that contain a lof of image augmentation parameters
        self.img_size = img_size # [640,640]
        self.name = name # name="train2017"
        self.json_file = json_file # '../../../../coco/annotations/instances_train2017.json'
        self.preproc = preproc # TrainTransform (used for data augmentation)
        self.augment = not no_aug # True/False, if no_aug is true, then Not no_aug is False, meaning augment is false
        self.tracking = tracking # True/False
        self.logger = logger # None, write down the log file?
        self.data_dir = self.opt.data_dir # image dir, '../../../../coco/images'
        self.batch_size = self.opt.batch_size # batch size

        # data augment params
        self.random_size = self.opt.random_size # [14,26], multi-size train: from 448(14*32) to 832(26*32), set None to disable it
        self.degrees = self.opt.degrees # rotate angle, 10
        self.translate = self.opt.translate # 0.1, used for mosaic augmentation
        self.scale = self.opt.scale # [0.1, 2], ???????
        self.shear = self.opt.shear # 2.0, used for mosaic augmentation
        self.perspective = self.opt.perspective # 0, used for mosaic augmentation
        self.mixup_scale = (0.5, 1.5) # used for mixup augmentation
        self.enable_mosaic = self.augment # True/False
        self.enable_mixup = self.opt.enable_mixup # True/False
        self.mosaic_prob = self.opt.mosaic_prob # 1, prob that uses mosaic augmentation
        self.mixup_prob = self.opt.mixup_prob # 1, prob that uses mixup augmentation

        #################
        # self.json_file = self.json_file.replace("train", "val")
        # self.name = self.name.replace("train", "val")
        #################
        assert os.path.isfile(self.json_file), 'cannot find {}'.format(self.json_file)
        print("==> Loading {} annotation {}".format(self.name, self.json_file))
        self.coco = COCO(self.json_file) # using cocotools to read json_file of training/val set
        self.ids = self.coco.getImgIds() # list that contains all training/val image ids
        self.num_samples = len(self.ids) # how many images on training/val set
        print("images number {}".format(self.num_samples))
        self.class_ids = sorted(self.coco.getCatIds()) # a list, [1, 2, ..., 90], 80 classes, some of them are None
        cats = self.coco.loadCats(self.coco.getCatIds()) # list, 80 length, (some of them are none, which has been removed), get each ids corresponding label, i.e. {'supercategory': 'food', 'id': 53, 'name': 'apple'}
        self.classes = [c["name"] for c in cats] # list contains those 80 labels, len: 80, only get those class names
        self.annotations = self._load_coco_annotations() # list (length is the train/val set length) contains all coordinates, classes, image ids, image names
        
       #  example: self.annotations[0] returns:
       #  (array([[359.17, 146.17, 470.62, 358.74,   3.  ],
       # [339.88,  22.16, 492.76, 321.89,   0.  ],
       # [471.64, 172.82, 506.56, 219.92,   0.  ],
       # [486.01, 183.31, 515.64, 217.29,   1.  ]]), (360, 640), '000000391895.jpg', 391895)
       # meaning this image has 4 objects, size is (360,640), name of the image and its ids.
        
        
        self.samples_shapes = [self.img_size for _ in range(self.num_samples)] # each image's height and width is stored here

        if 'val2017' == self.name:
            print("classes index:", self.class_ids)
            print("class names in dataset:", self.classes)

    def __len__(self):
        return self.num_samples

    def shuffle(self):
        np.random.shuffle(self.annotations)
        print("shuffle images list in {}".format(self.json_file))
        if self.logger:
            self.logger.write("shuffle {} images list...\n".format(self.json_file))

        if self.random_size is not None: #self.random_size = [14,26], multi-size train: from 448(14*32) to 800(25*32), set None to disable it
            self.samples_shapes = self.multi_shape()

    def multi_shape(self):
        size_factor = self.img_size[1] * 1. / self.img_size[0]

        multi_shapes = [] # [[448,448],[480,480],...,[800,800]]
        for size in list(range(*self.random_size)): # range([14,26]) -> 14,15,16...,25
            random_input_h, random_input_w = (int(32 * size), 32 * int(size * size_factor)) # 14*32, 14*1*32
            multi_shapes.append([random_input_h, random_input_w])
        print("multi size training: {}".format(multi_shapes))
        if self.logger:
            self.logger.write("multi size training: {}\n".format(multi_shapes))

        iter_num = int(np.ceil(self.num_samples / self.batch_size)) # ceil(0.2) = 1, ceil(-0.2) = 0
        samples_shapes = []
        rand_idx = len(multi_shapes) - 1  # initialize with max size, in case of out of memory during training
        for it in range(iter_num):
            if it != 0 and it % 10 == 0:
                rand_idx = np.random.choice(list(range(len(multi_shapes)))) # pick from [448,448] to [800,800] randomly, assign this h and w for the image
            for _ in range(self.batch_size):
                samples_shapes.append(multi_shapes[rand_idx])
        return samples_shapes
    
    
###############################################################################
    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes.keys():
            one_img_res = all_bboxes[image_id]
            for res in one_img_res:
                cls, conf, bbox = res[0], res[1], res[2]
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    'category_id': self.class_ids[self.classes.index(cls)],
                    'image_id': int(image_id),
                    'score': float(conf)})
        return detections

    def run_coco_eval(self, results, save_dir):
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'))
        coco_det = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        str_result = redirect_string.getvalue()
        ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large = coco_eval.stats[:6]
        print(str_result)
        return ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, str_result


################################################################################
    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids] # returns a list that contains all info about all images

    def load_anno_from_ids(self, id_): # for each image: load image annotations
        im_ann = self.coco.loadImgs(id_)[0] # load image annotations, including width, height, annotation_ids and file name
        width = im_ann["width"] # get each image width
        height = im_ann["height"] # get height
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False) # 获取加载对象所包含的所有标记信息（就是所有图片的Segmentation，即分割的坐标数据）
        annotations = self.coco.loadAnns(anno_ids) # len(list) = # of objects. a list that contains all annotations (segmentation and bbox coordinates, although we don't need segmentation info; class label, image id and id) 
        objs = []
        for obj in annotations: # get x,y coordinates for all objects
            x1 = np.max((0, obj["bbox"][0])) #obj['bbox'] contains each bbox's coordinates (shown in x1,y1, x2,y2 format)
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2] # convert from xywh to xyxy
                objs.append(obj)

        num_objs = len(objs) #get the number of objects in each image
        res = np.zeros((num_objs, 6 if self.tracking else 5)) # create an array where the shape is [num_objs, 5]
        for ix, obj in enumerate(objs): # for each bbox
            cls = self.class_ids.index(obj["category_id"]) # since category_id begins from 1, here convert it to begin from 0
            res[ix, 0:4] = obj["clean_bbox"] #put x1,y1,x2,y2 into res matrix
            res[ix, 4] = cls #put class label into res matrix
            if self.tracking:
                assert "tracking_id" in obj.keys(), 'cannot find "tracking_id" in your dataset'
                res[ix, 5] = obj['tracking_id']
                # print('errorrrrrrrr: replace tracking_id to cls')
                # res[ix, 5] = cls
        # Now res contains bbox and class info for this image, [[x1,y1,x2,y2,class], [x1,y1,x2,y2,class],[x1,y1,x2,y2,class]...]
        img_info = (height, width) 
        file_name = im_ann["file_name"]

        del im_ann, annotations
        # res: bbox and class coordinates, img_info: height and width of the image, file_name:image name, id_:image id
        return res, img_info, file_name, id_
###############################################################################


    def pull_item(self, index): #res: bbox coordinates and class, img_info: h and w of image, file_name: image file name, id_: image id
        res, img_info, file_name, id_ = self.annotations[index] # self.annotations: list that contains all label information of all images
        # load image and preprocess
        img_file = self.data_dir + "/" + self.name + "/" + file_name
        img = cv2.imread(img_file)
        assert img is not None, "error img {}".format(img_file)
        #img: input image, res: bbox coordinates and class, img_info: h and w of image, id_: image id
        return img, res.copy(), img_info, id_

    def close_random_size(self):
        self.samples_shapes = [self.img_size for _ in range(self.num_samples)]
        print("close multi-size training")

    def __getitem__(self, idx):
        # self.enable_mosaic： True/False, self.augment: True/False, random.random() < self.mosaic_prob: there are self.mosaic_prob% of the chance that mosaic is activated
        if self.enable_mosaic and self.augment and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_h, input_w = self.samples_shapes[idx]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # find another 3 additional image indices
            indices = [idx] + [random.randint(0, self.num_samples - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices): #img:input image, _labels_:corresponding bboxes and classes
                img, _labels, _, _ = self.pull_item(index) #pull the corresponding image and its info out from self.annotations
                h0, w0 = img.shape[:2]  # original image height and width
                scale = min(1. * input_h / h0, 1. * input_w / w0) # find scale factor
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR) # resize the image based on the scale factor
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0: 
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    # mosaic_img: an np array with (input_h * 2, input_w * 2, 3) size and all pixels filled with 114
                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    i_mosaic, xc, yc, w, h, input_h, input_w
                ) 
                # mosaic_img: an np array with (input_h * 2, input_w * 2, 3) size and all pixels filled with 114
                # i_mosaic: index from the indices list, used to find the corresponding image info
                # xc, yc:mosaic center
                # w, h: resized image height and weight
                #input_h, input_w: h and w of the image after applying multi_shape training

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # scale and move the label (xyxy)
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                # np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0]) # get all x1
                # np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1]) # get all y1
                # np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2]) # get all x2
                # np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3]) # get all y2
            
            
            # get the mosaic augmented image (no mixup yet!)
            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove
            
            
            # if we want to use mixup, then the following code will be excicuted
            if self.enable_mixup and not len(mosaic_labels) == 0 and random.random() < self.mixup_prob:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.samples_shapes[idx])
            img_info = (mosaic_img.shape[1], mosaic_img.shape[0])
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.samples_shapes[idx])
            return mix_img, padded_labels, img_info, -1
        else:
            img, label, img_info, img_id = self.pull_item(idx)
            img, label = self.preproc(img, label, self.samples_shapes[idx])
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_index = random.randint(0, self.__len__() - 1)
        img, cp_labels, _, _ = self.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            if self.tracking:
                tracking_id_labels = cp_labels[keep_list, 5:6].copy()
                labels = np.hstack((box_labels, cls_labels, tracking_id_labels))
            else:
                labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

# used in mixup augmentation
def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def get_mosaic_coordinate(mosaic_index, xc, yc, w, h, input_h, input_w):
    
    # TODO update doc
    # index0 to top left part of image
    
 
    # mosaic_index aka i_mosaic: index from the indices list [0,1,2,3], used to find the corresponding image info
    # xc, yc:mosaic center
    # w, h: resized image height and weight
    #input_h, input_w: h and w of the image after applying multi_shape training
    
    
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord





#%% not affiflited with dataset
def get_dataloader(opt, no_aug=False, logger=None, val_loader=True):
    do_tracking = opt.reid_dim > 0
    # train
    train_dataset = COCODataset(opt,
                                img_size=opt.input_size,
                                name='train2017',
                                json_file=opt.train_ann,
                                preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120,
                                                       tracking=do_tracking, augment=True),
                                no_aug=no_aug,
                                tracking=do_tracking,
                                logger=logger,
                                )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # do shuffle in dataset
        num_workers=opt.data_num_workers,
        pin_memory=True,
        drop_last=True
    )

    if not val_loader:
        return train_loader, None

    # val
    val_dataset = COCODataset(opt,
                              img_size=opt.test_size,
                              name='val2017',
                              json_file=opt.val_ann,
                              preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120,
                                                     tracking=do_tracking, augment=False),
                              no_aug=True,
                              tracking=do_tracking,
                              logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.data_num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader


def vis_inputs(inputs, targets, opt):
    from utils.util import label_color

    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    for b_i, inp in enumerate(inputs):
        target = targets[b_i]
        img = (((inp.transpose((1, 2, 0)) * opt.std) + opt.rgb_means) * 255).astype(np.uint8)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        gt_n = 0
        for t in target:
            if t.sum() > 0:
                if len(t) == 5:
                    cls, c_x, c_y, w, h = [int(i) for i in t]
                    tracking_id = None
                elif len(t) == 6:
                    cls, c_x, c_y, w, h, tracking_id = [int(i) for i in t]
                else:
                    raise ValueError("target shape != 5 or 6")
                bbox = [c_x - w // 2, c_y - h // 2, c_x + w // 2, c_y + h // 2] # convert from xyxy to xywh
                label = opt.label_name[cls]
                # print(label, bbox)
                color = label_color[cls]
                # show box
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                # show label and conf
                txt = '{}-{}'.format(label, tracking_id) if tracking_id is not None else '{}'.format(label)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color,
                              -1)
                cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)
                gt_n += 1

        print("img {}/{} gt number: {}".format(b_i, len(inputs), gt_n))
        cv2.namedWindow("input", 0)
        cv2.imshow("input", img)
        key = cv2.waitKey(0)

        if key == 27: # ESC's ASCII = 27, meaning if press ESC, it will quit
        # if key == ord('q'):
            exit()


def run_epoch(train_loader, e, opt):
    for batch_i, batch in enumerate(train_loader):
        inps, targets, img_info, ind = batch
        print("------------ epoch {} batch {}/{} ---------------".format(e, batch_i, len(train_loader)))
        print("batch img shape {}, target shape {}".format(inps.shape, targets.shape))
        if opt.show:
            vis_inputs(inps, targets, opt)
        if batch_i >= 21:
            break


def main():
    import sys
    sys.path.append('../')
    
    from config import opt

    # opt.dataset_path = '../../../../coco'
    
    opt.dataset_path = '../../../../converted_mask_data'
    # opt.dataset_path = "/media/ming/DATA1/dataset/coco2017"
    opt.train_ann = opt.dataset_path + "/annotations/instances_train2017.json"
    opt.val_ann = opt.dataset_path + "/annotations/instances_val2017.json"
    opt.data_dir = opt.dataset_path + "/images"

    opt.input_size = (640, 640)
    opt.test_size = (640, 640)
    opt.batch_size = 2
    opt.data_num_workers = 0  # 0
    opt.reid_dim = 0  # 128
    opt.show = True  # False
    opt.enable_mixup = True
    

    # # test __getitem__
    # train_dataset = COCODataset(opt,
    #                             img_size=opt.input_size,
    #                             name='train2017',
    #                             json_file=opt.train_ann,
    #                             preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120,
    #                                                    tracking=False, augment=True),
    #                             no_aug=False,
    #                             tracking=False,
    #                             logger=None,
    #                             )
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=False,  # do shuffle in dataset
    #     num_workers=opt.data_num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    


    train_loader, val_loader = get_dataloader(opt, no_aug=False)

    # train_loader = val_loader
    dataset_label = train_loader.dataset.classes
    
    opt.label_name = ['with_mask', 'mask_weared_incorrect', 'without_mask']
    
    assert opt.label_name == dataset_label, "your class_name != dataset's {} {}".format(opt.label_name, dataset_label)
    for e in range(1):
        train_loader.dataset.shuffle() # call the shuffle function, ramdonize the image height and width between [448, 448] to [800, 800]
        if e == 2:
            train_loader.dataset.enable_mosaic = True
        run_epoch(train_loader, e, opt)


if __name__ == "__main__":
    main()
