import numpy as np
import torch
import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader

# from maskrcnn_benchmark.modeling.rrpn.inference import make_rpn_postprocessor
from maskrcnn_benchmark.modeling.rrpn.loss import make_rpn_loss_evaluator

from maskrcnn_benchmark.modeling.rotated_box_coder import BoxCoder

# from maskrcnn_benchmark.structures.image_list import to_image_list
# from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.rrpn.anchor_generator import \
    make_anchor_generator as make_rrpn_anchor_generator, convert_rect_to_pts2, draw_anchors

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler


def get_feature_maps(image, feature_strides, device='cpu'):
    N, C, H, W = image.shape
    feature_maps = [torch.zeros(N,1,H//s,W//s, device=device) for s in feature_strides]
    return feature_maps

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    nx /= (xmax - xmin)
    return nx

if __name__ == '__main__':

    config_file = "./model_save/with_FPN/config.yml"
    #config_file = "./configs/mscoco/mscoco_miou_4x.yaml"
    # config_file = "./configs/pen_dataset/mrcnn_miou.yaml"
    try:
        cfg.merge_from_file(config_file)
    except KeyError as e:
        print(e)
    cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.0#  1.0
    cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0
    cfg.INPUT.PIXEL_MEAN = [0,0,0]
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.MODEL.RPN.ANCHOR_STRIDE = (32,)
    cfg.freeze()

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )

    device = 'cpu'

    anchor_generator = make_rrpn_anchor_generator(cfg)
    num_anchors = anchor_generator.num_anchors_per_location()

    print(num_anchors)

    box_coder = BoxCoder(weights=None) #cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = make_rpn_loss_evaluator(cfg, box_coder)

    start_iter = 0
    count_vizu = 0
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        feature_maps = get_feature_maps(images.tensors, cfg.MODEL.RPN.ANCHOR_STRIDE)

        anchors = anchor_generator.forward(images, feature_maps)
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        anchors_cnt = [len(a) for a in anchors]

        labels, regression_targets, matched_gt_ids, _ \
            = loss_evaluator.prepare_targets(anchors, targets)

        sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        total_pos = sampled_pos_inds.numel()
        total_neg = sampled_neg_inds.numel()

        cumu_cnt = 0

        regression_targets = torch.cat(regression_targets, dim=0)#.cpu().numpy()

        start_gt_idx = 0
        for ix, t in enumerate(targets):
            matched_gt_ids[ix] += start_gt_idx
            start_gt_idx += len(t)

        matched_gt_ids = torch.cat(matched_gt_ids)
        # matched_gt_ious = torch.cat(matched_gt_ious)

        img_tensors = images.tensors

        # print(total_pos, total_neg)

        # pos_regression_targets = regression_targets[sampled_pos_inds]
        # print(np.rad2deg(pos_regression_targets[:,-1]))
        device = matched_gt_ids.device

        # pos_matched_gt_ids = matched_gt_ids[sampled_pos_inds]
        # pos_matched_gt_ious = matched_gt_ious[sampled_pos_inds]
        # label_idxs = [torch.nonzero(pos_matched_gt_ids == x).squeeze() for x in range(start_gt_idx)]
        # label_weights = torch.zeros_like(pos_matched_gt_ids, dtype=torch.float32)
        # MAX_GT_NUM = 10
        # label_cnts = [min(MAX_GT_NUM, nz.numel()) for nz in label_idxs]
        # total_pos = sum(label_cnts)
        # for x in range(start_gt_idx):
        #     nz = label_idxs[x]
        #     nnn = nz.numel()
        #     if nnn <= MAX_GT_NUM:
        #         if nnn > 0:
        #             label_weights[nz] = total_pos / nz.numel()
        #         continue
        #     # top_iou_ids = torch.sort(pos_matched_gt_ious[nz], descending=True)[1][:MAX_GT_NUM]
        #     # inds = nz[top_iou_ids]
        #     label_weights[inds] = total_pos / MAX_GT_NUM

        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       = total_pos / label_cnts.to(dtype=torch.float32)
        
        for ix,cnt in enumerate(anchors_cnt):
            gt = targets[ix]

            gt_bbox = np.round(gt.bbox.cpu().numpy())
            # gt_mask_instance = gt.get_field("masks")
            # gt_polygons = [p.polygons for p in gt_mask_instance.instances.polygons]
            # for gx,gtp in enumerate(gt_polygons):
            #     gtp = [p.view(-1, 2).cpu().numpy() for p in gtp]
            #     gt_polygons[gx] = gtp
            gt_rrects = get_boxlist_rotated_rect_tensor(gt, "masks").cpu().numpy()

            img_t = img_tensors[ix]
            inds = sampled_pos_inds[cumu_cnt < sampled_pos_inds]
            inds = inds[inds < (cumu_cnt+cnt)]

            pos_anchors = anchors[ix][inds - cumu_cnt]
            reg_targets = regression_targets[inds]
            reg_target_gt_inds = matched_gt_ids[inds]
            # reg_target_gt_ious = matched_gt_ious[inds]

            cumu_cnt += cnt

            anchor_rrects = pos_anchors.get_field("rrects")
            rr = anchor_rrects.cpu().numpy()
            # print(rr)

            assert reg_targets.shape == anchor_rrects.shape

            # reg_targets[gt_135, -1] = 0

            # reg_targets[:, -1] = reg_targets_angles
            # print(np.rad2deg(reg_targets[:, -1]))
            # print(reg_target_gt_inds)
            # print(reg_target_gt_ious)
            proposals = box_coder.decode(reg_targets, anchor_rrects).cpu().numpy()

            img = img_t.cpu().numpy()
            img = np.transpose(img, [1,2,0]).copy()
            # img = img[:,:,::-1]  # rgb to bgr
            # img = normalize(img, 0, 1)
            img *= 255
            img = img.astype(np.uint8)

            img2 = img.copy()
            img3 = img.copy()

            for bbox in gt_bbox:
                cv2.rectangle(img, tuple(bbox[:2].astype(int)), tuple(bbox[2:].astype(int)), (0,0,255))
            img = draw_anchors(img, gt_rrects)
            img2 = draw_anchors(img2, rr)
            img3 = draw_anchors(img3, proposals)

            print('image,', count_vizu )
            cv2.imwrite( 'anchor_visu/gt'+str(count_vizu)+'.jpg', img)
            cv2.imwrite('anchor_visu/matching anchors '+str(count_vizu) +'.jpg', img2)
            cv2.imwrite('anchor_visu/anchor proposals'+str(count_vizu) +'.jpg', img3)
            count_vizu+=1
        if count_vizu >5 :
            break
