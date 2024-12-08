import gc
import matplotlib.pyplot as plt
import torch
import numpy as np
from multiprocessing import Pool
import time
from sklearn import linear_model
from scipy.optimize import minimize
from sklearn import linear_model

# from src.utils.utils import *
def error_func(coeffs, ip):
    a, b = coeffs
    return a ** 2 * b ** 2 * ip[0, 0] + a * b ** 2 * ip[0, 1] + a ** 2 * b * ip[0, 2] + a * b * ip[0, 3] \
    + a * b ** 2 * ip[1, 0] + b ** 2 * ip[1, 1] + b * a * ip[1, 2] + b * ip[1, 3] \
    + a ** 2 * b * ip[2, 0] + a * b * ip[2, 1] + a ** 2 * ip[2, 2] + a * ip[2, 3] \
    + a * b * ip[3, 0] + b * ip[3, 1] + a * ip[3, 2] + ip[3, 3]

def solve(ip):
    res = minimize(error_func, x0=[0.5, 0.5], args=(ip,), bounds=[(0, 1), (0, 1)], method='L-BFGS-B')
    return res

def round_outward(array):
    negative_mask = array < 0
    array = np.abs(array)
    array = np.ceil(array)
    array[negative_mask] *= -1
    return array

def round_outward(array):
    negative_mask = array < 0
    array = np.abs(array)
    array = np.ceil(array)
    array[negative_mask] *= -1
    return array

def generate_neighbor_offset(num_sample, sample_std):
    # assert (sample_std * 2 + 1) ** 2 > num_sample
    iteration = 0
    while (True):
        iteration += 1
        neighbor_offset = np.random.randn(num_sample - 1, 2) * sample_std
        neighbor_offset = round_outward(neighbor_offset).astype(int)
        values, counts = np.unique(neighbor_offset, return_counts=True, axis=0)
        if counts.max() < 2 and [0, 0] not in values:
            break
    neighbor_offset = np.concatenate([neighbor_offset, np.zeros((1, 2))], axis=0)
    return neighbor_offset.astype(int)


class Demo:
    def __init__(self, imgs, ft, img_size, upsample_factor):
        self.ft = ft # NCHW
        self.imgs = imgs
        self.upsample_factor = upsample_factor
        if type(img_size) not in [tuple, list]:
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size # [H, W]

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):

        fig, axes = plt.subplots(1, 2, figsize=(fig_size*2, fig_size))

        plt.tight_layout()

        for i in range(2):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        # num_channel = self.ft.size(1)
        # cos = nn.CosineSimilarity(dim=1)

        with torch.no_grad():
            upsample_factor = self.upsample_factor # only upsample src
            src_map = torch.nn.functional.upsample(self.ft[0], size=(int(self.ft[0].size(2) * upsample_factor), int(self.ft[0].size(3) * upsample_factor)), mode='bilinear', align_corners=True)
            src_ft = src_map.permute(2, 3, 0, 1).flatten(0, 1) # 1, C, H, W -> H, W, 1, C -> num_src, 1, C
            _, C, H_src, W_src = src_map.size()
            target_map = self.ft[1].permute(0, 2, 3, 1) # 1, H, W, C 
            _, H_tgt, W_tgt, C = target_map.size()
            print("source map size", src_ft.size())
            print("target map size", target_map.size())
            v00 = target_map[..., :-1, :-1, :].flatten(1, 2) # 1, num_tgt, C
            v10 = target_map[..., 1:, :-1, :].flatten(1, 2)
            v01 = target_map[..., :-1, 1:, :].flatten(1, 2)
            v11 = target_map[..., 1:, 1:, :].flatten(1, 2)
            cell_center_ft = (v00 + v10 + v01 + v11) / 4 # 1, num_tgt, C
            num_src = src_ft.size(0)
            # find the top k closest grid cells to search in
            l2_dist = (cell_center_ft - src_ft).norm(dim=-1, p=2) # num_src, num_tgt
            top_k_to_search = 30
            min_idx_topk = l2_dist.argsort(dim=1)[:, :top_k_to_search] # num_src, k
            min_error_mtx = np.zeros((num_src, top_k_to_search))
            coeff_mtx = np.zeros((num_src, top_k_to_search, 2))
            del l2_dist, cell_center_ft
            top_k_inner_mtx = np.zeros((num_src, top_k_to_search, 4, 4))

            for k in range(top_k_to_search):
                start = time.time()
                kth_min_idx = min_idx_topk[:, k]
                # find corresponding corners of cells
                w1 = (v00 - v10 - v01 + v11)[0, kth_min_idx] # num_src, C
                w2 = (v10 - v11)[0, kth_min_idx] # num_src, C
                w3 = (v01 - v11)[0, kth_min_idx] # num_src, C
                w4 = v11[0, kth_min_idx] - src_ft[:, 0, :] # num_src, C
                w_list = torch.stack([w1, w2, w3, w4], dim=1) # num_src, 4, C
                inner_product_mtx = torch.bmm(w_list, w_list.transpose(1, 2)).cpu().numpy() # num_src, 4, 4
                # w_list = [w1, w2, w3, w4]
                # inner_product_mtx = [] # 4, 4, num_src
                # for i in range(len(w_list)):
                #     row = []
                #     for j in range(len(w_list)):
                #         xij = (w_list[i] * w_list[j]).sum(-1).cpu().numpy() # num_src
                #         row.append(xij)
                #         torch.cuda.empty_cache()
                #     inner_product_mtx.append(row)
                # inner_product_mtx = np.array(inner_product_mtx).transpose(2, 0, 1) # num_src, 4, 4
                top_k_inner_mtx[:, k, :, :] = inner_product_mtx
                del w1, w2, w3, w4, w_list
            top_k_inner_mtx = top_k_inner_mtx.reshape(-1, 4, 4) # num_src * top_k_to_search, 4, 4

        start = time.time()
        with Pool(16) as p:
            results = p.map(solve, [top_k_inner_mtx[i] for i in range(num_src * top_k_to_search)])
        # results = []
        # for ik in range(num_src * top_k_to_search):
        #     results.append(solve(top_k_inner_mtx[ik]))
        print("solving took", time.time() - start)
            
        ik_to_results_idx = np.arange(num_src * top_k_to_search).reshape(num_src, top_k_to_search)
        for i in range(num_src):
            for k in range(top_k_to_search):              
                coeff_mtx[i, k] = results[ik_to_results_idx[i, k]].x
                min_error_mtx[i, k] = results[ik_to_results_idx[i, k]].fun
        print("solving + assignment took", time.time() - start)

        # find the best k, then find index of best match
        best_k = np.argmin(min_error_mtx, axis=1) # num_src
        best_idx = min_idx_topk.cpu().numpy()[np.arange(num_src), best_k] # num_src            
        best_coeff_mtx = coeff_mtx[np.arange(num_src), best_k] # num_src, 2
        del v00, v10, v01, v11, src_ft, target_map

        grid_x_tgt, grid_y_tgt = np.meshgrid(np.arange(W_tgt - 1), np.arange(H_tgt - 1), indexing="xy")
        grid_x_tgt, grid_y_tgt = grid_x_tgt.reshape(-1), grid_y_tgt.reshape(-1) # x and y coordinate of upper left corner of each cell, num_src
        base_x, base_y = grid_x_tgt[best_idx], grid_y_tgt[best_idx] # num_src
        offset_y, offset_x = 1 - best_coeff_mtx[:, 0], 1 - best_coeff_mtx[:, 1] # num_src
        correspondences_x, correspondencs_y = base_x + offset_x, base_y + offset_y # num_src

        # for each point, sample some neighboring points
        num_neighbors = 5
        sample_std = 1

        grid_x_src, grid_y_src = np.meshgrid(np.arange(W_src), np.arange(H_src), indexing="xy") # H_src, W_src
        grid_xy_src = np.stack([grid_x_src, grid_y_src], axis=2).reshape(num_src, 2).astype(int) # num_src, 2
        neighbor_xy = np.zeros((num_src, num_neighbors, 2)).astype(int) # num_src, num_neighbors, 2
        predicted_coeffs = np.zeros((num_src, 2, 2)) # num_src, 2, 2
        predicted_biases = np.zeros((num_src, 2)) # num_src, 2
        confidence_scores = np.zeros(num_src)
        inlier_ratios = np.zeros(num_src)

        
        ij_to_src_idx = np.arange(num_src).reshape(H_src, W_src)
        start = time.time()
        for i in range(num_src):
            neighbor_offset = generate_neighbor_offset(num_neighbors, sample_std) # num_neighbors, 2
            neighbor_xy_i = grid_xy_src[i] + neighbor_offset # num_neighbors, 2
            neighbor_xy_i[:, 0] = np.clip(neighbor_xy_i[:, 0], 0, W_src - 1)
            neighbor_xy_i[:, 1] = np.clip(neighbor_xy_i[:, 1], 0, H_src - 1)
            # fit ransac
            neighbor_idx = ij_to_src_idx[neighbor_xy_i[:, 1], neighbor_xy_i[:, 0]] # num_neighbors
            Yx, Yy = correspondences_x[neighbor_idx], correspondencs_y[neighbor_idx]
            Yi = np.stack([Yx, Yy], axis=1) # num_neighbors, 2
            ransac = linear_model.RANSACRegressor().fit(neighbor_xy_i, Yi)
            predicted_coeffs[i] = ransac.estimator_.coef_
            predicted_biases[i] = ransac.estimator_.intercept_
            
            inlier_mask = ransac.inlier_mask_
            # those two should be equal
            # print(Yi[inlier_mask].mean(axis=0) - np.array(ransac.estimator_.coef_) @ neighbor_xy_i[inlier_mask].mean(axis=0))
            # print(ransac.estimator_.intercept_)
            # outlier_mask = np.logical_not(inlier_mask)
            confidence_scores[i] = ransac.score(neighbor_xy_i[inlier_mask], Yi[inlier_mask])
            inlier_ratios[i] = sum(inlier_mask) / num_neighbors
            # save to matrix
            neighbor_xy[i, :, :] = neighbor_xy_i
            
        print("fitting ransac took", time.time() - start)
        # print("generating neighbors took", time.time() - start)

        score_thres = 0.9 # [-inf, 1]
        inlier_ratio_thres = 0.75


        def onclick(event):
            src_scale_h, src_scale_w = (self.img_size[0][1] - 1) / (H_src - 1), (self.img_size[0][0] - 1) / (W_src - 1)
            tgt_scale_h, tgt_scale_w = (self.img_size[1][1] - 1) / (H_tgt - 1), (self.img_size[1][0] - 1) / (W_tgt - 1)
            if event.inaxes == axes[0]:
                x, y = event.xdata, event.ydata
                x_grid, y_grid = int(x / src_scale_w), int(y / src_scale_h)
                src_idx = ij_to_src_idx[y_grid, x_grid]
                # print("input idx", x_grid, y_grid)
                # print("matched idx", base_x[src_idx], base_y[src_idx])
                # print("offset", offset_x[src_idx], offset_y[src_idx])
                axes[0].clear()
                axes[0].imshow(self.imgs[0])
                axes[0].axis('off')
                axes[0].scatter(x, y, c='r', s=scatter_size)
                axes[0].scatter(neighbor_xy[src_idx][:, 0] * src_scale_w, neighbor_xy[src_idx][:, 1] * src_scale_h, c='b', s=scatter_size)
                axes[0].set_title('source image')
                neighbor_idx = ij_to_src_idx[neighbor_xy[src_idx, :, 1], neighbor_xy[src_idx, :, 0]] # num_neighbors
                max_yx = np.stack([correspondencs_y[neighbor_idx], correspondences_x[neighbor_idx]], axis=1) # num_neighbors, 2
                # print(max_xy.shape, predicted_coeffs.shape, predicted_biases.shape)
                predicted_correspondence_xy = np.matmul(neighbor_xy[src_idx], predicted_coeffs[src_idx].T) + predicted_biases[src_idx]
                print("confidence: ", confidence_scores[src_idx], "inliers: ", inlier_ratios[src_idx])
                U, S, Vt = np.linalg.svd(predicted_coeffs[src_idx])
                angle1 = np.arctan2(U[1, 0], U[0, 0])
                angle2 = np.arctan2(Vt[1, 0], Vt[0, 0])
                print("affine transforamtion rotations are: ", S, angle1 / np.pi * 180, angle2 / np.pi * 180, ((angle1 - angle2) / np.pi * 180 + 360) % 360)

                axes[1].clear()
                axes[1].imshow(self.imgs[1])
                # heatmap = cos_map[i-1]
                # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                # axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                # del cos_map
                # del heatmap
                axes[1].axis('off')
                axes[1].scatter(max_yx[:, 1] * tgt_scale_w, max_yx[:, 0] * tgt_scale_h, c='r', s=scatter_size)
                # axes[1].scatter(predicted_correspondence_xy[:, 0] * tgt_scale_w, predicted_correspondence_xy[:, 1] * tgt_scale_h, c='g', s=scatter_size)
                axes[1].set_title('target image')
                fig.canvas.draw()
                gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        