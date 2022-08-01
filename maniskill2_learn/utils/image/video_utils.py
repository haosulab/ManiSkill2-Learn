import cv2, numpy as np, imageio
import os.path as osp


def video_to_frames(filename, output_dir=None):
    video = cv2.VideoCapture(filename)
    images = []
    count = 0
    success = True
    while success:
        success, image = video.read()
        if success:
            if output_dir is None:
                images.append(image)
            else:
                cv2.imwrite(osp.join(output_dir, f"frame_{count}.jpg"), image)
                count += 1
    return np.stack(images, axis=0)[..., ::-1]


def grid_images(images, max_col_num=4):
    total_num = ((len(images) + max_col_num - 1) // max_col_num) * max_col_num
    images = list(images) + [np.zeros_like(images[0]) for i in range(total_num - len(images))]
    # print(len(images), max_col_num, total_num)
    ret = []
    for i in range(total_num // max_col_num):
        row_i = []
        for j in range(max_col_num):
            k = i * max_col_num + j
            row_i.append(images[k])
        ret.append(np.concatenate(row_i, axis=2))
    ret = np.concatenate(ret, axis=1)
    return ret


def put_names_on_image(images, names):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for image, name in zip(images, names):
        for i in range(len(image)):
            # print(image.shape, name, font)
            cv2.putText(image[i], name, (10, image.shape[1] - 10), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            # import matplotlib.pyplot as plt
            # plt.imshow(image[i])
            # plt.show()
            # cv2.imwrite(path + 'pillar_text.jpg', im)


def concat_videos(filenames, output_filename, names=None, max_col_num=4, fps=10):
    images = [video_to_frames(filename) for filename in filenames]
    num = np.max([image.shape[0] for image in images])
    images = [
        np.concatenate(
            [
                image,
            ]
            + [image[-1:] for i in range(num - image.shape[0])],
            axis=0,
        )
        for image in images
    ]

    if names is not None:
        assert len(names) == len(images)
        put_names_on_image(images, names)

    images = grid_images(images, max_col_num)
    images_to_video(images, output_filename, fps=fps)


def images_to_video(images, filebane, fps=10):
    writer = imageio.get_writer(filebane, fps=fps)
    for im in images:
        writer.append_data(im)
    writer.close()
