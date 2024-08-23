import cv2
import torch
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from modules.depth_anything_v2.dpt import DepthAnythingV2

class ParallaxGenerator():
    def __init__(self, input_image, encoder='vits', max_parallax=15):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = encoder
        self.model = DepthAnythingV2(**model_configs[self.encoder])
        self.model.load_state_dict(torch.load(f'modules/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
        self.model = self.model.to(DEVICE).eval()
        self.img = input_image
        self.max_parallax = max_parallax
        self.height, self.width = input_image.shape[:2]

    def generate(self):
        self.depth = self.model.infer_image(self.img)
        self.depth = (self.depth - self.depth.min()) / (self.depth.max() - self.depth.min()) * 255
        self.depth = self.depth.astype(np.uint8)
        self.left_img , self.right_img = self.createParallax()
        self.left_img = self.current_parallax(self.left_img)
        self.right_img = self.current_parallax(self.right_img)

    def createParallax(self) -> cv2.Mat:
        segmented_img = self.get_mask_segment_all()
        relation = self.relation_label_depth(segmented_img)
        left_img = np.zeros((self.height, self.width,3), np.uint8)
        right_img = np.zeros((self.height, self.width,3), np.uint8)
        # 距離画像を参照しながら左右画像の画素をずらす
        for label, value in relation:
            label = int(label)
            parallax = int((value - relation[:,1].min()) / (relation[:,1].max() - relation[:,1].min()) * self.max_parallax)
            mask = (segmented_img == label).astype(np.uint8) * 255
            # print(f"self.img: dtype={self.img.dtype},size={self.img.shape}; mask: dtype={mask.dtype},size={mask.shape}")

            tx, ty = parallax, 0
            # print(f"label={label}, tx={tx}, ty={ty}")
            temp_target = cv2.bitwise_and(self.img, self.img, mask=mask)
            mv_mat_left = np.float32([[1, 0, tx],[0, 1, ty]])
            mv_mat_right = np.float32([[1, 0, -tx],[0, 1, ty]])
            affine_left = cv2.warpAffine(temp_target, mv_mat_left, (self.width, self.height))
            affine_right = cv2.warpAffine(temp_target, mv_mat_right, (self.width, self.height))
            left_img = np.where(affine_left != 0, affine_left, left_img)
            right_img = np.where(affine_right != 0, affine_right, right_img)
        return left_img, right_img

    def get_mask_segment_all(self):
        flat_image = self.depth.reshape((-1,1))
        quantile = 0.2
        n_samples = 500
        bandwidth = estimate_bandwidth(flat_image, quantile=quantile, n_samples=n_samples)
        # print(f"Calculated bandwidth: {bandwidth}")
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(flat_image)
        labels = ms.labels_
        segmented_image = labels.reshape(self.depth.shape)
        return segmented_image

    def relation_label_depth(self, segmented_img) -> np.array:
        label_list = np.unique(segmented_img)
        relation = np.zeros((len(label_list), 2), dtype=np.int32)
        # ラベルごとに深度の中央値を求める
        for label in label_list:
            mask = (segmented_img == label).astype(np.uint8) * 255
            domain = cv2.bitwise_and(self.depth, self.depth, mask=mask)
            pixel_list = np.unique(domain)[1:]
            median = np.median(pixel_list)
            relation[label] = [label, median]
        # ラベリングされた分割領域を整列して前後関係を把握する
        sorted_relation = relation[relation[:, 1].argsort()]
        return sorted_relation

    def current_parallax(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hole_mask = (gray_img==0).astype(np.uint8) * 255
        current = cv2.inpaint(img, hole_mask, 3, cv2.INPAINT_TELEA)
        return current

    def getStereogram(self, swapLR=False, split=False):
        if split:
            return self.left_img, self.right_img
        margin_height = self.height // 10
        margin_width = self.width // 20
        dot = np.ones((margin_height, self.width, 3), dtype=np.uint8) * 255
        cv2.circle(dot, (dot.shape[1]//2, dot.shape[0]//2), margin_height//3, (0,0,0), -1)
        dot_blank = np.ones((margin_height, margin_width,3), dtype=np.uint8) * 255
        dot_zone = cv2.hconcat([dot, dot_blank, dot])
        blank = np.ones((self.height, margin_width, 3), dtype=np.uint8) * 255
        if swapLR:
            dst_img = cv2.hconcat([self.right_img, blank, self.left_img])
        else:
            dst_img = cv2.hconcat([self.left_img, blank, self.right_img])
        dst_img = cv2.vconcat([dst_img, dot_zone])
        return dst_img
