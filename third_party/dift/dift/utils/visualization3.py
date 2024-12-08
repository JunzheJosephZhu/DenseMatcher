import gc
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn import svm

class Demo:

    def __init__(self, imgs, ft, img_size, dummy=0):
        self.ft = ft # NCHW
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size # list of tuples

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft[0].size(1)
        cos = nn.CosineSimilarity(dim=1)

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():

                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

                    src_ft = self.ft[0]
                    src_ft = nn.Upsample(size=self.img_size[0][::-1], mode='bilinear')(src_ft)
                    src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

                    # train an SVM
                    start = time.time()
                    _, channels, h, w = self.ft[0].shape
                    upsample_factor_h = self.img_size[0][1] / h
                    upsample_factor_w = self.img_size[0][0] / w
                    labels = np.zeros((h, w), dtype=bool)
                    labels[int(np.round(y / upsample_factor_h)), int(np.round(x / upsample_factor_w))] = True
                    labels = labels.flatten() # [h, w]
                    features = self.ft[0].reshape(channels, -1).cpu().numpy().T # [h*w, channels]
                    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
                    clf.fit(features, labels)
                    print("training SVM used", time.time() - start)
                    
                    del src_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    trg_ft = nn.Upsample(size=self.img_size[1][::-1], mode='bilinear')(self.ft[1])
                    cos_map = cos(src_vec, trg_ft).cpu().numpy()  # N, H, W
                    _, channels, tgt_h, tgt_w = trg_ft.shape
                    similarities = clf.decision_function(trg_ft.reshape(channels, -1).cpu().numpy().T)
                    sorted_ix = np.argsort(-similarities)

                    del trg_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].scatter(x, y, c='r', s=scatter_size)
                    axes[0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        # max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                        max_yx = np.unravel_index(sorted_ix[0], cos_map[i-1].shape)
                        axes[i].clear()

                        heatmap = cos_map[i-1]
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[i].imshow(self.imgs[i])
                        # axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[i].axis('off')
                        axes[i].scatter(max_yx[1].item(), max_yx[0].item(), c='r', s=scatter_size)
                        axes[i].set_title('target image')
                    fig.canvas.draw()


                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()