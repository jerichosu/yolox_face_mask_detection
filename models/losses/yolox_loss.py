# -*- coding: utf-8 -*-
# @Time    : 21-7-20 20:01



import numpy as np
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOXLoss(nn.Module): #reid_dim: used for tracking
    def __init__(self, label_name, reid_dim=0, id_nums=None, strides=[8, 16, 32], in_channels=[256, 512, 1024]):
    # def __init__(self, num_classes, reid_dim=0, id_nums=None, strides=[8, 16, 32], in_channels=[256, 512, 1024]):
        super().__init__()

        self.n_anchors = 1
        self.label_name = label_name
        self.num_classes = len(self.label_name)
        # self.num_classes = num_classes
        self.strides = strides
        self.reid_dim = reid_dim

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(in_channels)

        # if self.reid_dim > 0:
        #     assert id_nums is not None, "opt.tracking_id_nums shouldn't be None when reid_dim > 0"
        #     assert len(id_nums) == self.num_classes, "num_classes={}, which is different from id_nums's length {}" \
        #                                              "".format(self.num_classes, len(id_nums))
        #     # scale_trainable = True
        #     # self.s_det = nn.Parameter(-1.85 * torch.ones(1), requires_grad=scale_trainable)
        #     # self.s_id = nn.Parameter(-1.05 * torch.ones(1), requires_grad=scale_trainable)

        #     self.reid_loss = nn.CrossEntropyLoss(ignore_index=-1)
        #     self.classifiers = nn.ModuleList()
        #     self.emb_scales = []
        #     for idx, (label, id_num) in enumerate(zip(self.label_name, id_nums)):
        #         print("{}, tracking label name: '{}', tracking_id number: {}, feat dim: {}".format(idx, label, id_num,
        #                                                                                            self.reid_dim))
        #         self.emb_scales.append(np.math.sqrt(2) * np.math.log(id_num - 1))
        #         self.classifiers.append(nn.Linear(self.reid_dim, id_num))

    def forward(self, preds, targets, imgs=None): #preds: [[B, (which class,x,y,h,w + 80), 80, 80],[B, 85, 40, 40],[B, 85, 20, 20]]
        outputs, origin_preds, x_shifts, y_shifts, expanded_strides = [], [], [], [], []

        for k, (stride, p) in enumerate(zip(self.strides, preds)): # strides = [8, 16, 32]
                                        # languages = ['Java', 'Python', 'JavaScript']
                                        # versions = [14, 3, 6]
                                        # result = zip(languages, versions)
                                        # print(list(result))
                                        # # Output: [('Java', 14), ('Python', 3), ('JavaScript', 6)]
            pred, grid = self.get_output_and_grid(p, k, stride, p.dtype) #pred: adjusted x,y,h,w
            # pred:torch.Size([1, 6400, 85]), grid: torch.Size([1, 6400, 2])
            outputs.append(pred) # [[1, 6400, 85], [1, 1600, 85], [1, 400, 85]]
            x_shifts.append(grid[:, :, 0]) # [[1, 6400],[1, 1600],[1, 400]] for x  torch.Size([1, 6400]), since grid is [1, 6400, 2], meaning there are 2 [1, 6400] stack together
            y_shifts.append(grid[:, :, 1]) # [[1, 6400],[1, 1600],[1, 400]] for y  torch.Size([1, 6400])
            expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride).type_as(p)) # expanded_strides[0].shape: torch.Size([1, 6400]) filled with all 8 (stride)
            # expanded_strides: [[1,6400 filled with all 8], [1,1600 filled with all 16], [1,400 filled with all 32]]
            if self.use_l1:
                reg_output = p[:, :4, :, :]
                batch_size, _, hsize, wsize = reg_output.shape
                reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                reg_output = (reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4))
                origin_preds.append(reg_output.clone())

        outputs = torch.cat(outputs, 1) # torch.Size([1, 6400 + 1600 + 400, 85])
        total_loss, iou_loss, conf_loss, cls_loss, l1_loss, reid_loss, num_fg = self.get_losses(imgs, # None
                                                                                                x_shifts, # [[1, 6400],[1, 1600],[1, 400]] for x
                                                                                                y_shifts, # [[1, 6400],[1, 1600],[1, 400]] for y
                                                                                                expanded_strides, # [[1,6400 filled with all 8], [1,1600 filled with all 16], [1,400 filled with all 32]]
                                                                                                targets, # torch.Size([1, num_of_objects_in_image, 5])
                                                                                                outputs, # torch.Size([1, 6400 + 1600 + 400, 85])
                                                                                                origin_preds, #None
                                                                                                dtype=preds[0].dtype) # float32 or float16 for mixed precision???

        losses = {"loss": total_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "iou_loss": iou_loss}
        if self.use_l1:
            losses.update({"l1_loss": l1_loss})
        # if self.reid_dim > 0:
        #     losses.update({"reid_loss": reid_loss})
        losses.update({"num_fg": num_fg}) # add number of foreground into the dictionary 
        return losses

    def get_output_and_grid(self, p, k, stride, dtype): #p: prediction from each scale, torch.Size([1, 85, 80, 80]), k:0,1,2, stride: 8,16,32, dtype:float32 (float16 with mixed precision maybe?)        
        p = p.clone() #？？？？
        grid = self.grids[k] #  self.grids = [tensor([0.]), tensor([0.]), tensor([0.])], k = 0,1,2
        batch_size, n_ch, hsize, wsize = p.shape

        if grid.shape[2:4] != p.shape[2:4] or grid.device != p.device: #device:cpu or gpu, p.shape[2:4] = torch.size[80,80]
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)]) #xv:0,1,2//0,1,2//0,1,2//..., yv:0,0,0//1,1,1//2,2,2...
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype).to(p.device) # torch.Size([1, 1, 80, 80, 2])
            self.grids[k] = grid # torch.Size([1, 1, 80, 80, 2])

        pred = p.view(batch_size, self.n_anchors, n_ch, hsize, wsize) # torch.Size([1, 1, 85, 80, 80])
        pred = (pred.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)) #torch.Size([1, 6400, 85])
        grid = grid.view(1, -1, 2) # torch.Size([1, 6400, 2])
        pred[..., :2] = (pred[..., :2] + grid) * stride # stride:[8,16,32], grid: torch.Size([1, 6400, 2]), pred[..., :2]:torch.Size([1, 6400, 2]), adjust the x,y to the center of each cell
        pred[..., 2:4] = torch.exp(pred[..., 2:4]) * stride # stride:[8,16,32], pred[..., 2:4]: adjust h,w of each box
        return pred, grid

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, targets, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]  # [batch, 8400, 4:(x,y,h,w)]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, 8400, 1:(fifth index)]
        cls_preds = outputs[:, :, 5:self.num_classes + 5]  # [batch, 8400, n_cls (6~85, represents prob for each class)]
        # if self.reid_dim > 0:
        #     reid_preds = outputs[:, :, self.num_classes + 5:]  # [batch, h*w, 128]
        # targets (input label): torch.Size([1, num_of_objects_in_image, 5:(x,y,h,w,class)])
        assert targets.shape[2] == 6 if self.reid_dim > 0 else 5 # make sure targets.shape[2] == 5
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects on the image

        total_num_anchors = outputs.shape[1] # 8400, outputs.shape: [batch, 8400, 85]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all] # torch.Size([1, 8400]), tensor([[ 0.,  1.,  2., ...80, repeated for 6400, 0,1,2...40, repeated for 1600, 0,1,2 ..., 20, repeated for 400]])
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all] # torch.Size([1, 8400]), tensor([[ 0,0,0...0(80 ZEROS),1,1,1...1(80 ONES),2,2,2...(80 TWOS) ... 80,80,80...,(80 eightys)0, 0,0,0...0(40 zeros)..., 0,0,0...(20 zeros)]])
        expanded_strides = torch.cat(expanded_strides, 1) # torch.Size([1, 8400]), tensor([[ 8.,  8.,  8.,  ..., 32., 32., 32.]])
        if self.use_l1: #false
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        # reid_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]): # outputs.shape: [1, 8400, 85], 1 is batch size
            num_gt = int(nlabel[batch_idx]) # how many real bboxes in each image
            num_gts += num_gt #total number of real bboxes for this batch
            if num_gt == 0: # if this image has no bbox
                cls_target = outputs.new_zeros((0, self.num_classes)) # self.num_classes = 80, new_zeros: https://pytorch.org/docs/stable/generated/torch.Tensor.new_zeros.html
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                reid_target = outputs.new_zeros((0, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else: # targets.shape: torch.Size([1, 5, 5]), 5 bboxes, each has class,x,y,h,w (5 entries)
                gt_classes = targets[batch_idx, :num_gt, 0] # get classes for each bboxes in each image
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5] # get each bbox coordinates (x,y,h,w) in each image
                # if self.reid_dim > 0:
                #     gt_tracking_id = targets[batch_idx, :num_gt, 5]
                bboxes_preds_per_image = bbox_preds[batch_idx, :, :] # get predicted bbox coordinates (x,y,h,w) from network output, # bbox_preds.shape: [batch, 8400, 4:(x,y,h,w)]
                # bboxes_preds_per_image.shape: torch.Size([8400, 4])
                try:
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        # noqa
                        batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                        cls_preds, bbox_preds, obj_preds, targets, imgs,
                    ) 
                    
                
                # gt_matched_classes:       which class is assigned for each matched cell
                
                # fg_mask:      all 8400 cells, those who gets assigned cells are true, the rest of those are false.
                #         note fg_mask has been updated here in the dynamic_k_matching function, shape: torch.size([8400])
                
                # pred_ious_this_matching:      iou of each predictions with true bbox that has the same class 
                #     example: the gt_matched_classes = tensor([6., 4., 4., 6., 3., 7., 3.]), meaning the first cell is assigned
                #     as class 6...
                #     then compute the iou with gt_classes: tensor([3., 6., 8., 7., 4.]) that contains class 6, which is the second entrices
                #     the result is tensor([0.3554, 0.2381, 0.1325, 0.5419, 0.4119, 0.2541, 0.4223])
                
                # matched_gt_inds:
                #     example: we have 7 assigned predictions, and gt has 5 bboxes, each bbox has class, 
                #     they are: tensor([3., 6., 8., 7., 4.]), then inds means each prediction is assigned 
                #     to one class, if matched_gt_inds = tensor([1, 4, 4, 1, 0, 3, 0]), that means the first assigned 
                #     prediction is assigned as class 6 (since gt_classes[1] = 6) 
                
                # num_fg (num_fg_img):       how many assigned cells
                    

                except RuntimeError: # above is GPU mode, if GPU mode is not ok, then use CPU mode here, code are same
                    print(traceback.format_exc())
                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        # noqa
                        batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                        cls_preds, bbox_preds, obj_preds, targets, imgs, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img # add matched cells for each image together (how many matched cells we have for a batch)
                # convert match target into the format we want, to compute loss (these are assgined targets! NOT predictions!)
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes) * pred_ious_this_matching.unsqueeze(-1) # convert to [# foreground, # of classes]
                obj_target = fg_mask.unsqueeze(-1) # add one dimension, shape [8400,1]
                reg_target = gt_bboxes_per_image[matched_gt_inds] # actual assigned cells: torch.Size([# of assigned cells, 4 (x,y,h,w)])

                # if self.reid_dim > 0:
                #     reid_target = gt_tracking_id[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target) # include eaach image's class information into a list
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
            # if self.reid_dim > 0:
            #     reid_targets.append(reid_target)

        cls_targets = torch.cat(cls_targets, 0) # convert from list back to tensor
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        # if self.reid_dim > 0:
        #     reid_targets = torch.cat(reid_targets, 0).type(torch.int64)
        # COMPUTE LOSS HERE
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg if self.use_l1 else 0.

        reid_loss = 0.
        # if self.reid_dim > 0:
        #     reid_feat = reid_preds.view(-1, self.reid_dim)[fg_masks]
        #     cls_label_targets = cls_targets.max(1)[1]
        #     for cls in range(self.num_classes):
        #         inds = torch.where(cls == cls_label_targets)
        #         if inds[0].shape[0] == 0:
        #             continue
        #         this_cls_tracking_id = reid_targets[inds]
        #         this_cls_reid_feat = self.emb_scales[cls] * F.normalize(reid_feat[inds])

        #         reid_output = self.classifiers[cls](this_cls_reid_feat)
        #         reid_loss += self.reid_loss(reid_output, this_cls_tracking_id)

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + reid_loss
        fg_r = torch.tensor(num_fg / max(num_gts, 1), device=outputs.device, dtype=dtype)
        return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, reid_loss, fg_r
        # loss: total loss
        # reg_weight * loss_iou: bbox loss
        # loss_obj: objective loss
        # loss_cls: class loss
        # loss_l1: None
        # reid_loss: None (for tracking)
        # fg_r: num_fg/num_gt (number of assigned cells / number of gt) like a ratio
    
    

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
            bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
            cls_preds, bbox_preds, obj_preds, targets, imgs, mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        # fg_mask.shape: torch.Size([8400]), union,//// is_in_boxes_and_center: torch.Size([5, # of trues(cells where center and bbox overlapped for each bbox)])
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask] # [100(# of union cells), 4], leave those union cells' coordinates prediction
        cls_preds_ = cls_preds[batch_idx][fg_mask] # [100(# of union cells), 80], leave those union cells' class prediction
        obj_preds_ = obj_preds[batch_idx][fg_mask] # [100(# of union cells), 1], leave those union cells' objective prediction
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0] # 100 (# of union cells)

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # compute the iou between union cells and true bboxs for each image
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        
        # ex: original image has 5 bbox, each bbox has 1 class, so total # of class is 5, if total # of class is 80, then after onehot it will be [0 0 0 0 1(fifth is 1) 0 0 ..., total 80], for each class
        # shape: [5, # of union cells(used for computing cost), 80(# of class after onehot, coco is 80)]
        gt_cls_per_image = ( # gt_classes: classes for each bboxes in each image, torch.size([5]) --> 5 bboxes, each has its own classes
            F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor,
                                                                                                1)) # convert to onehot type
       
        # compute loss for iou, will be used for computing cost as shown in paper's equation (1)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8) # shape : torch.Size([5, # of union cells])
        # 5: 5 bboxes in image, each compute loss with all # of union cells, so each has [1, # of union cells] cost, 5 will have [5, # of union cells] shape loss
        
        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
        
        
        # cls_preds_: # of union cell's class prediction, each has 80 class predictions, so shape is [# of union cells, 80]
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                        cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid() * obj_preds_.unsqueeze(0).repeat(
                    num_gt, 1, 1).sigmoid())
            # after this cls_preds_ shape: [5, # of union cells, 80 (# classes)]
            # compute class loss, used for compute cost in equation (1), shape: [5, # of union cells] (each true class needs to compute loss with all other (# of union cells), and there are 5 bboxes)
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_
        
        
        # cost = class_loss + lambda*iou_loss, shown in paper equation (1) https://arxiv.org/pdf/2107.08430.pdf 
        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center))
        # shape: [5, # of union cells], each true bbox compute cost with all other # of union cells, there are 5 bboxes, so shape is [5, # of union cells]
        
        
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
            
            
        '''
        num_fg: how many mached cells
        gt_matched_classes: which class is assigned for each matched cell
        pred_ious_this_matching: iou of each predictions with true bbox that has the same class 
            example: the gt_matched_classes = tensor([6., 4., 4., 6., 3., 7., 3.]), meaning the first cell is assigned
            as class 6...
            then compute the iou with gt_classes: tensor([3., 6., 8., 7., 4.]) that contains class 6, which is the second entrices
            the result is tensor([0.3554, 0.2381, 0.1325, 0.5419, 0.4119, 0.2541, 0.4223])
        
        matched_gt_inds:
            example: we have 7 assigned predictions, and gt has 5 bboxes, each bbox has class, 
            they are: tensor([3., 6., 8., 7., 4.]), then inds means each prediction is assigned 
            to one class, if matched_gt_inds = tensor([1, 4, 4, 1, 0, 3, 0]), that means the first assigned 
            prediction is assigned as class 6 (since gt_classes[1] = 6) 
            
        # fg_mask: all 8400 cells, those who gets assigned cells are true, the rest of those are false.
                note fg_mask has been updated here in the dynamic_k_matching function, shape: torch.size([8400])
        '''
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg
    
    
    # gt_bboxes_per_image: x,y,h,w for each box in each image
    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        expanded_strides_per_image = expanded_strides[0] #expanded_strides.shape: [1, 8400], expanded_strides[0].shape: [8400] 
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image #[0*8, 1*8, 2*8,...39*8, ... 0*16,1*16...19*16, ...0*32, 1*32...19*32]
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image # [0*8, 0*8,...0*8, 1*8, 1*8,...1*8, ... 39*8, 39*8, 0*16, 0*16,..., 0*32, ...]
        x_centers_per_image = ((x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = ((y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))
        # get left top and bottom right coordinates of each bbox from the (x,y,h,w) of each bbox coordinate
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        )
        # compute which cell (for all 8400 cells) contains GT
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0 # those who has values > 0 are GTs
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + \
                                center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all # compute the union
        # print(torch.count_nonzero(is_in_boxes_anchor))
        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]) # compute intersection
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        '''
        cost: cost that is computed to determine k gts, torch.Size([5, # of union cells]), equation (1)
        pair_wise_ious: iou between # of union cells and each true bboxs for each image, torch.Size([5, # of union cells]) (each cell needs to compute ious with all union cells, there are 5 bboxes, so [5, # of union cells])
        gt_classes: get classes for each true bboxes in each image [0(cat),9(dog),7(cars),0,1...this is before onehot!], torch.size([5]), since 5 boxes, each has 1 class
        num_gt: how many bboxes in each image, here we have 5 bboxes, so num_gt = 5
        fg_mask: all 8400 cells, those who is true means they are union cells, see torch.count_nonzero(fg_mask), shape: torch.Size([8400])
        '''
        matching_matrix = torch.zeros_like(cost) # shape: [5, # of union cells], all entries are zero

        ious_in_boxes_matrix = pair_wise_ious # iou that is computed for select the k value
        # n_candidate_k = 10
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1)) # ious_in_boxes_matrix.size(1): # of union cells, make sure to select at least 10 values
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1) # compute each row's top 10 biggest ious, shape: [5, 10]
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1) # for each bbox, add top k ious and floor it(6.68 -> 6), shape: [5], each bbox has 1 sum of ious, there are 5 bboxes
        for gt_idx in range(num_gt): # num_gt = 5
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False) #pick dynamic_k (based on iou) smallest cost's index for each row (each bboxes)
            matching_matrix[gt_idx][pos_idx] = 1.0 # assign those small cost with value 1 on that zero matrix (matrix shape: [5, # of union cells], with all zeros, only those with small cost will be filled with 1)

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0) # add them ROW BY ROW!!! --> [5, # OF UNION CELLS] -> [# OF UNION CELLS]
        if (anchor_matching_gt > 1).sum() > 0: # if any  entries in anchor_matching_gt has 2 or bigger values
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0) 
            # anchor_matching_gt > 1: [True(if >1), False,...total # of union cell entries], 
            # cost[:, anchor_matching_gt > 1]: for each row, find values of those with trues, shape: [5, # of those which contains True]
            # ex: aa = torch.tensor([1,2,3,4,5]), bb = torch.tensor([True,True,False,False,True]) --> aa[bb] = tensor([1, 2, 5])
            # torch.min(cost[:, anchor_matching_gt > 1], dim=0): find the overall smallest cost and its indix 
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0 # EACH COLUMN ONLY HAS ONE 1! only assign those that has the smallest cost to be 1 for each column
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0 # sum [5, # of union cells] by column, Note after processing the matching_matrix, each column has one 1. shape:[# of union cells]
        num_fg = fg_mask_inboxes.sum().item() # compute hoe many trues we have (gts used to compute loss)

        fg_mask[fg_mask.clone()] = fg_mask_inboxes # fg_mask_inboxes hape: [# of union cells] with true or false
        # fg_mask: now only those cells with trues were placed
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds] # assign class for each matched cell

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        '''
        num_fg: how many mached cells
        gt_matched_classes: which class is assigned for each matched cell
        pred_ious_this_matching: iou of each predictions with true bbox that has the same class 
            example: the gt_matched_classes = tensor([6., 4., 4., 6., 3., 7., 3.]), meaning the first cell is assigned
            as class 6...
            then compute the iou with gt_classes: tensor([3., 6., 8., 7., 4.]) that contains class 6, which is the second entrices
            the result is tensor([0.3554, 0.2381, 0.1325, 0.5419, 0.4119, 0.2541, 0.4223])
        
        matched_gt_inds:
            example: we have 7 assigned predictions, and gt has 5 bboxes, each bbox has class, 
            they are: tensor([3., 6., 8., 7., 4.]), then inds means each prediction is assigned 
            to one class, if matched_gt_inds = tensor([1, 4, 4, 1, 0, 3, 0]), that means the first assigned 
            prediction is assigned as class 6 (since gt_classes[1] = 6) 
        '''
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
    

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = area_i / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


if __name__ == "__main__":
    import sys
    sys.path.append('../../')
    from config import opt

    # torch.manual_seed(opt.seed)
    # opt.reid_dim = 128  # 0 # used for tracking problem
    opt.batch_size = 1
    opt.num_classes = 20
    
    max_obj_num = 5 #number of objects in a labelled image, 5 means there are 5 bboxes 
    # dummy_input = [torch.rand([opt.batch_size, 85 + opt.reid_dim, i, i]) for i in [64, 32, 16]]
    # opt.num_classes + 5: number of class + class, x, y, h, w
    dummy_input = [torch.rand([opt.batch_size, (opt.num_classes + 5) + opt.reid_dim, i, i]) for i in [80,40,20]]
    dummy_target = torch.rand([opt.batch_size, max_obj_num, 6 if opt.reid_dim > 0 else 5]) * 50  # [bs, max_obj_num, 6]      
    dummy_target[:, :, 0] = torch.randint(10, (opt.batch_size, max_obj_num), dtype=torch.int64) # first column of dummy_target
    
    # if opt.reid_dim > 0:
    #     dummy_target[:, :, 5] = torch.randint(20, (opt.batch_size, 3), dtype=torch.int64)
    #     opt.tracking_id_nums = []
    #     for dummy_id_num in range(50, len(opt.label_name) + 50):
    #         opt.tracking_id_nums.append(dummy_id_num)

    # THIS IS THE ORIGINAL VERSION
    # yolox_loss = YOLOXLoss(label_name=opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums)
    
    
    # yolox_loss = YOLOXLoss(label_name=opt.label_name) # label_name: used to obtain # of classes, can be modified in code!
    yolox_loss = YOLOXLoss(num_classes=opt.num_classes)
    
    
    print('input shape:', [i.shape for i in dummy_input])
    print("target shape:", dummy_target, dummy_target.shape)

    loss_status = yolox_loss(dummy_input, dummy_target)
    for l in loss_status:
        print(l, loss_status[l])
