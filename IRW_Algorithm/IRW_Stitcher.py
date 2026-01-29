from utils import *
from utils.recursive import RecursiveDivider
import sys
from denseMatcher import matchany_loftr
from warp import meshgrid, solveMeshWarping, calcProj, warpingMesh
from seam import sigmoidSimilarity, seamFinding, seamBlending
import skimage
import time
import os
from PIL import Image
import cv2

class IRW_Stitcher:
    def __init__(self, opt):
        self.opt = opt
        self.n = 0 
        self.unit_w, self.unit_h = self.opt.resize

    def load_image(self, path):
        try:
            src = cv2.imread(path, cv2.IMREAD_COLOR)
            assert src is not None, f"File not found: {path}"
            src = src[:, :, ::-1] 
            src = cv2.resize(src, (self.unit_w, self.unit_h))
            return src
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
            return None

    def recursive(self, imgdir):
        if isinstance(imgdir, list):
            if len(imgdir) == 2:
                return self.IRW_Stitching(self.recursive(imgdir[0]), self.recursive(imgdir[1]))
            else:
                return self.recursive(imgdir[0])
        else:
            src = cv2.imread(imgdir[0], cv2.IMREAD_COLOR)
            assert src is not None, print(f'No such directory exists:{imgdir[0]}')
            src = src[:, :, ::-1]
            src = cv2.resize(src, dsize=(self.unit_w, self.unit_h))
            try:
                # process stitching
                dst = cv2.imread(imgdir[1], cv2.IMREAD_COLOR)[:, :, ::-1]
                dst = cv2.resize(dst, dsize=(self.unit_w, self.unit_h))
                return self.IRW_Stitching(src, dst)
            except:
                return src

    def process(self, imgdir, mask):
        unit_start = time.perf_counter()
        if self.opt.print_n: print(f'processing {self.n + 1} thread...')
        result = self.recursive(imgdir)
        if isinstance(mask, str):
            mask_img = cv2.imread(mask, cv2.IMREAD_COLOR)[:, :, ::-1]
            mask_img = cv2.normalize(mask_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            h, w, _ = mask_img.shape
            result = cv2.resize(result, dsize=(w, h))
            result *= mask_img
        elif isinstance(mask, list):
            mask_img = cv2.imread(mask[self.n], cv2.IMREAD_COLOR)[:, :, ::-1]
            mask_img = cv2.normalize(mask_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            h, w, _ = mask_img.shape
            result = cv2.resize(result, dsize=(w, h))
            result *= mask_img
        else:
            pass
            os.makedirs(self.opt.saveroot, exist_ok=True)
            save_dir = os.path.join(self.opt.saveroot, self.opt.savename +
                                    str(self.n).zfill(5) + '.' + self.opt.savefmt)
            cv2.imwrite(save_dir, result[:, :, ::-1])
            if self.opt.saveprint: print(f'{self.n+1} image saved -> {save_dir}')
        if self.opt.image_show > self.n or self.opt.image_show == -1:
            result = Image.fromarray(result)
            result.show()
        if self.opt.unit_time: print(f'{self.n + 1} image time spending: {time.perf_counter() - unit_start:4f}s.')
        self.n += 1

    @staticmethod
    def call_dataset(fname, root=None):
        file = open(fname, 'r')
        data = file.readlines()
        target_stack = []
        for d in data:
            imgname = d.strip().strip('\n').split(' ')
            if root is not None:
                target_stack.append([os.path.join(root, name) for name in imgname])
            else:
                target_stack.append(imgname)
        return target_stack

    def call_mask(self):
        if self.opt.mask_dir is None:
            return None
        try:  # only one mask
            mask = self.opt.mask_dir
        except:  # mask text file
            mask = self.call_dataset(self.opt.mask_dir, root=self.opt.mask_root)
        return mask

    def thread_choice(self):
        # mask setting
        mask = self.call_mask()
        # divider instance
        divider = RecursiveDivider()
        # two image stitching
        if None not in [self.opt.img1, self.opt.img2]:
            data = divider.list_divide([self.opt.img1, self.opt.img2])
            self.process(data, mask)
        # multi image stitching
        elif self.opt.imgs is not None:
            data = divider.list_divide(self.opt.imgs)
            self.process(data, mask)
        # image (root + txt list merging) or (absolute) path stitching
        elif None not in [self.opt.imgroot, self.opt.imglist]:
            datalist = self.call_dataset(self.opt.imglist, root=self.opt.imgroot)
            for data in datalist:
                data = divider.list_divide(data)
                self.process(data, mask)
        # error
        else:
            print('please enter input options.')

    def IRW_Stitching(self, src, dst):
        rgb_imgs = [dst,src]
        image0 = dst
        image1 = src
        # Dense Matcher
        mkpts0_matchany, mkpts1_matchany = matchany_loftr(image0=image0, image1=image1)
        point_pairs = [mkpts0_matchany, mkpts1_matchany]
        layered_points1, layered_points2, _ = featureTriangle(point_pairs[0], point_pairs[1], 20)
        H, _, _ = calcHomography([layered_points1[0], layered_points2[0]], ransac=False)
        _, points1, points2 = calcHomography(point_pairs, ransac=True)
        [Out_rect, In_rect, rect0, rect1, offset] = calcRect(rgb_imgs[0], rgb_imgs[1], H)

        # IRW Algorithm
        h = 40
        [ori_vertices, x_num, y_num] = meshgrid(Out_rect[0], Out_rect[1], Out_rect[2], Out_rect[3], h, h)
        proj_vertices = ori_vertices
        # initialize point weights
        points_weight_error = np.ones(points1.shape[0], np.float32)
        last_points_weight = points_weight_error.copy()
        # iteratively update mesh warping model and weights
        for i in range(100):
            points_weight = points_weight_error
            if ((np.max(np.abs(points_weight - last_points_weight)) < 0.01) & (i > 0)) | (i == 99):
                break
            proj_vertices = solveMeshWarping(points1, points2, x_num, y_num, h, Out_rect[0], Out_rect[1], ori_vertices,
                                            H, 5, 0.1, points_weight)
            points_weight_error = calcProj(points1, points2, proj_vertices, x_num, h, Out_rect[0], Out_rect[1], 0.005)
            last_points_weight = points_weight
        warped_imgs = warpingMesh(rgb_imgs, proj_vertices, x_num, y_num, h, Out_rect, rect0)

        # compute metrics

        # metric_res1 = canvas_to_img1(warped_imgs[1], rgb_imgs[0].shape, offset).astype(np.float32)
        # metric_mask1 = canvas_to_mask(warped_masks[1], rgb_imgs[0].shape, offset)
        # metric_mask1 = metric_mask1.astype(np.uint8)
        # metric_res2 = canvas_to_img1(warped_imgs[0], rgb_imgs[0].shape, offset).astype(np.float32)

        # psnr = skimage.metrics.peak_signal_noise_ratio(
        #     metric_res1 * np.expand_dims(metric_mask1, axis=2),
        #     metric_res2 * np.expand_dims(metric_mask1, axis=2),
        #     data_range=255
        # )

        # ssim = skimage.metrics.structural_similarity(
        #     metric_res1 * np.expand_dims(metric_mask1, axis=2), 
        #     metric_res2 * np.expand_dims(metric_mask1, axis=2), 
        #     data_range=255, 
        #     channel_axis=-1)

        # image blending
        similarity_map0 = sigmoidSimilarity(warped_imgs, warped_masks[0] * warped_masks[1])
        seam_masks = seamFinding(similarity_map0, warped_masks)
        result = seamBlending(warped_imgs, seam_masks)
        return result