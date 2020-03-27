# -*- coding: utf8 -*-
from pathlib import Path
import time
import tensorflow as tf


import tensorlayer as tl
from model import Generator
from utils import read_all_imgs

class SRService:

    def __init__(self, type):
        if type == 'x2':
            self.upscale = 1
            self.checkpoint_dir = "checkpoint/x2"
            self.size_init = size_init=(256, 256, 3)
        elif type == 'x4':
            self.upscale = 2
            self.checkpoint_dir = "checkpoint/x4"
            self.size_init = size_init=(128, 128, 3)

        else:
            raise RuntimeError('unknown xx type')

        self.save_dir = "output"
        tl.files.exists_or_mkdir(self.save_dir)

        self.size_init = size_init

        self.t_image = tf.placeholder('float32', [None, size_init[0], size_init[1], size_init[2]], name='input_image')

        self.net_g = Generator(self.t_image, is_train=False, reuse=False, upscale=self.upscale)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        tl.layers.initialize_global_variables(self.sess)
        tl.files.load_and_assign_npz(sess=self.sess, name=self.checkpoint_dir + '/g_srgan.npz', network=self.net_g)

    def evaluate_file(self, input_path):
        try:
            assert Path(input_path).is_file(), "input file does not exist"

            valid_lr_img_list = [input_path]
            valid_lr_imgs = read_all_imgs(valid_lr_img_list, n_threads=32)
            init_lr_img = valid_lr_imgs[0]
            size_init = init_lr_img.shape

            assert size_init == self.size_init, "input image size mismatch with model"

            imid = valid_lr_img_list[0]
            imid = Path(imid).stem
            valid_lr_img = valid_lr_imgs[0]

            valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

            ###======================= EVALUATION =============================###
            start_time = time.time()
            out = self.sess.run(self.net_g.outputs, {self.t_image: [valid_lr_img]})
            print("took: %4.4fs" % (time.time() - start_time))

            print(f"input LR size: {size_init} / generated HR size: {out.shape}") # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)

            outpath = str(Path(self.save_dir).joinpath(f'{imid}_sr.png'))
            tl.vis.save_image(out[0], outpath)
            print(f"save output to file {outpath}")

            return outpath
        except Exception as e:
            print('Error: {}'.format(e.__class__.__name__))
            print(e)

if __name__ == '__main__':
    sr_service = SRService(type='x2')

    while True:

        path = input("please enter the input image path:")
        sr_service.evaluate_file(path)