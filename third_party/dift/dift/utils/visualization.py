import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import logging
import os
if os.path.exists('tmp.log'):
    os.remove('tmp.log')
# create logger with 'spam_application'
logger = logging.getLogger('visualization')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('tmp.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

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
                    logger.info(f"{self.ft[0]}, {self.ft[1]}")
                    logger.info(f"feature shapes: {self.ft[0].shape}, {self.ft[1].shape}")
                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))


                    src_ft = self.ft[0]
                    src_ft = nn.Upsample(size=self.img_size[0][::-1], mode='bilinear')(src_ft)
                    src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1
                    logger.info(f"src_ft shape: {src_ft.shape}, src_vec shape: {src_vec.shape}")

                    del src_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    trg_ft = nn.Upsample(size=self.img_size[1][::-1], mode='bilinear')(self.ft[1])
                    cos_map = cos(src_vec, trg_ft).cpu().numpy()  # N, H, W

                    del trg_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].scatter(x, y, c='c', s=scatter_size)
                    axes[0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        logger.info(f"cos map: {cos_map}")
                        logger.info(f"cos map shape: {cos_map[i-1].shape}")
                        max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                        logger.info(f"max_yx: {max_yx}")
                        axes[i].clear()

                        heatmap = cos_map[i-1]
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[i].imshow(self.imgs[i])
                        # axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[i].axis('off')
                        axes[i].scatter(max_yx[1].item(), max_yx[0].item(), c='b', s=scatter_size)
                        axes[i].set_title('target image')
                    fig.canvas.draw()


                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()