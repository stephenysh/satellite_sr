#! /usr/bin/python
# -*- coding: utf8 -*-
import time
import tensorflow as tf
import tensorlayer as tl
from model import Generator
from utils import read_all_imgs

def evaluate(args):
    if args.type == 'x2':
        upscale = 1
        checkpoint_dir = "checkpoint/x2"
    elif args.type == 'x4':
        upscale = 2
        checkpoint_dir = "checkpoint/x4"

    else:
        raise RuntimeError('unknown xx type')

    save_dir = "output"
    tl.files.exists_or_mkdir(save_dir)

    imid_lr_dir = '/tmp/pycharm_project_100/90032/'

    valid_lr_img_list = sorted(tl.files.load_file_list(path=imid_lr_dir, regx='.*.jpg', printable=False))
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=imid_lr_dir, n_threads=32)
    init_lr_img = valid_lr_imgs[0]
    size_init = init_lr_img.shape

    t_image = tf.placeholder('float32', [None, size_init[0], size_init[1], size_init[2]], name='input_image')

    net_g = Generator(t_image, is_train=False, reuse=False, upscale=upscale)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    for index in range(0, len(valid_lr_img_list)):
        imid = valid_lr_img_list[index]
        imid = imid[-14:-4]
        valid_lr_img = valid_lr_imgs[index]

        valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))

        print("LR size: %s /  generated HR size: %s" % (
        size_init, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/' + imid + '_sr.png')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default='x2', choices=['x2', 'x4'])

    args = parser.parse_args()

    evaluate(args)

