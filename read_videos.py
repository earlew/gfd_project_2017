from ffmpy import FFmpeg
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
import pickle
import copy

debug_here = Tracer()

video_dir = "/Users/ewilson2011/Desktop/Lab_videos/"
run_name_list = ['july14_run1', 'july17_run1', 'july20_run1', 'july21_run1', 'july24_run1', 'july25_run1']


def get_image_settings(run_name):
    run_params = {}

    if run_name == 'july14_run1':
        run_params['a'] = 2
        run_params['b'] = 2
        run_params['x_r'] = slice(100 * run_params['a'], 1200 * run_params['a'])
        run_params['y_r'] = slice(200 * run_params['b'], 330 * run_params['b'])
        run_params['tube_x0_p'] = 10 * run_params['a']  # pixels
        run_params['tube_y0_p'] = 95 * run_params['b']  # pixels
        run_params['tube_h_p'] = 37 * run_params['b']  # pixels
        run_params['ri_thresh'] = 0.01  # upper bound of red channel intensity within FW layer
        run_params['ri_thresh_minL'] = 20  # number of consecutive pixels to satisfy run_params['ri_thresh'] criteria
        run_params['base_offset_pix'] = -6 * run_params['b']  # pixels
        run_params['frames_dir'] = 'july14_run/rotated_rz_frames/'
        run_params['run_dir'] = 'july14_run/'
        run_params['max_w_p'] = 1000 * run_params['a']  # pixels
        run_params['im_dt'] = 5  # time interval of each image
        run_params['t_switch'] = np.array([30, 180, 260, 370, 480, 860])  # instances when pump speed changes
        run_params['t_eq'] = run_params['t_switch'][1:] - 10
        run_params['pump_set'] = np.array([350, 300, 250, 200, 150])
        run_params['pump_flux'] = get_vol_flux(run_params['pump_set'])
        run_params['x_cut_off_cm'] = 64
        run_params['ds'] = 33
        run_params['tilt_fix'] = 0.5

    elif run_name == 'july17_run1':

        run_params['a'] = 2
        run_params['b'] = 2
        run_params['x_r'] = slice(50 * run_params['a'], 1200 * run_params['a'])
        run_params['y_r'] = slice(150 * run_params['b'], 280 * run_params['b'])
        run_params['tube_x0_p'] = 10 * run_params['a']  # pixels
        run_params['tube_y0_p'] = 103 * run_params['b']  # pixels
        run_params['tube_h_p'] = 90  # pixels
        run_params['ri_thresh'] = 0.02  # upper bound of red channel intensity within FW layer
        run_params['ri_thresh_minL'] = 20  # number of consecutive pixels to satisfy run_params['ri_thresh'] criteria
        run_params['base_offset_pix'] = -5 * run_params['b']  # pixels
        run_params['frames_dir'] = 'july17_run/rotated_rz_frames/'
        run_params['run_dir'] = 'july17_run/'
        run_params['max_w_p'] = 1000 * run_params['a']  # pixels
        run_params['im_dt'] = 5  # time interval of each image
        run_params['t_switch'] = np.array([200, 390, 635, 840, 975, 1085, 1200])  # instances when pump speed changes
        run_params['t_eq'] = run_params['t_switch'][1:] - 10
        run_params['t_eq'][-1] = 1200
        run_params['pump_set'] = np.array([150, 200, 250, 300, 350, 400])
        run_params['pump_flux'] = get_vol_flux(run_params['pump_set'])
        run_params['x_cut_off_cm'] = 68
        run_params['tilt_fix'] = -0.2
        run_params['ds'] = 33

    elif run_name == 'july20_run1':

        run_params['a'] = 2
        run_params['b'] = 2
        run_params['x_r'] = slice(40 * run_params['a'], 1190 * run_params['a'])
        run_params['y_r'] = slice(150 * run_params['b'], 280 * run_params['b'])
        run_params['tube_x0_p'] = 10 * run_params['a']  # pixels
        run_params['tube_y0_p'] = 95 * run_params['b']  # pixels
        run_params['tube_h_p'] = 90  # pixels
        run_params['ri_thresh'] = 0.12  # upper bound of red channel intensity within FW layer
        run_params['ri_thresh_minL'] = 20  # number of consecutive pixels to satisfy run_params['ri_thresh'] criteria
        run_params['base_offset_pix'] = -8 * run_params['b']  # pixels
        run_params['frames_dir'] = 'july20_run/rotated_rz_frames/'
        run_params['run_dir'] = 'july20_run/'
        run_params['max_w_p'] = 1000 * run_params['a']  # pixels
        run_params['im_dt'] = 5  # time interval of each image
        run_params['t_switch'] = np.array([0, 100, 365, 8 * 60 + 20, 15 * 60 + 35,
                                           17 * 60 + 55, 21 * 60 + 25])  # instances when pump speed changes
        run_params['t_eq'] = run_params['t_switch'][1:] - 5
        run_params['pump_set'] = np.array([300, 150, 300, 150, 300, 150])
        run_params['pump_flux'] = get_vol_flux(run_params['pump_set'])
        run_params['x_cut_off_cm'] = 68
        run_params['tilt_fix'] = -0.1
        run_params['ds'] = 33

    elif run_name == 'july21_run1':

        run_params['a'] = 2
        run_params['b'] = 2
        run_params['x_r'] = slice(10 * run_params['a'], 1160 * run_params['a'])
        run_params['y_r'] = slice(150 * run_params['b'], 280 * run_params['b'])
        run_params['tube_x0_p'] = 25  # pixels
        run_params['tube_y0_p'] = 95 * run_params['b']  # pixels
        run_params['tube_h_p'] = 80  # pixels
        run_params['ri_thresh'] = 0.12  # upper bound of red channel intensity within FW layer
        run_params['ri_thresh_minL'] = 50  # number of consecutive pixels to satisfy run_params['ri_thresh'] criteria
        run_params['base_offset_pix'] = -5 * run_params['b']  # pixels
        run_params['frames_dir'] = 'july21_run/rotated_rz_frames/'
        run_params['run_dir'] = 'july21_run/'
        run_params['max_w_p'] = 1000 * run_params['a']  # pixels
        run_params['im_dt'] = 5  # time interval of each image
        run_params['t_switch'] = np.array([0, 8*60, 17*60 + 10, 19*60 + 15, 23*60+50])  # instances when pump speed changes
        run_params['t_eq'] = run_params['t_switch'][1:] - 5
        run_params['pump_set'] = np.array([150, 250, 350, 150])
        run_params['pump_flux'] = get_vol_flux(run_params['pump_set'])
        run_params['x_cut_off_cm'] = 68
        run_params['ds'] = 33
        run_params['tilt_fix'] = 0.1


    elif run_name == 'july24_run1':
        run_dir = run_name[:-1]
        run_params['a'] = 2
        run_params['b'] = 2
        run_params['x_r'] = slice(5 * run_params['a'], 1155 * run_params['a'])
        run_params['y_r'] = slice(150 * run_params['b'], 280 * run_params['b'])
        run_params['tube_x0_p'] = 30  # pixels
        run_params['tube_y0_p'] = 95 * run_params['b']  # pixels
        run_params['tube_h_p'] = 77  # pixels
        run_params['ri_thresh'] = 0.12  # upper bound of red channel intensity within FW layer
        run_params['ri_thresh_minL'] = 50  # number of consecutive pixels to satisfy run_params['ri_thresh'] criteria
        run_params['base_offset_pix'] = -5 * run_params['b']  # pixels
        run_params['frames_dir'] = '%s/rotated_rz_frames/' % run_dir
        run_params['run_dir'] = '%s/' % run_dir
        run_params['max_w_p'] = 1000 * run_params['a']  # pixels
        run_params['im_dt'] = 5  # time interval of each image
        run_params['t_switch'] = np.array([0, 145, 305, 545, 715, 930, 1080])  # instances when pump speed changes
        run_params['t_eq'] = run_params['t_switch'][1:] - 5
        run_params['pump_set'] = np.array([350, 250, 200, 150, 100, 125])
        run_params['pump_flux'] = get_vol_flux(run_params['pump_set'])
        run_params['x_cut_off_cm'] = 68
        run_params['ds'] = 33./2
        run_params['tilt_fix'] = -0.3

    elif run_name == 'july25_run1':
        run_dir = run_name[:-1]
        run_params['a'] = 2
        run_params['b'] = 2
        run_params['x_r'] = slice(5 * run_params['a'], 1155 * run_params['a'])
        run_params['y_r'] = slice(175 * run_params['b'], 305 * run_params['b'])
        run_params['tube_x0_p'] = 70  # pixels
        run_params['tube_y0_p'] = 185  # pixels
        run_params['tube_h_p'] = 75  # pixels
        run_params['ri_thresh'] = 0.12  # upper bound of red channel intensity within FW layer
        run_params['ri_thresh_minL'] = 50  # number of consecutive pixels to satisfy run_params['ri_thresh'] criteria
        run_params['base_offset_pix'] = -5 * run_params['b']  # pixels
        run_params['frames_dir'] = video_dir + '%s/rotated_rz_frames/' % run_dir
        run_params['run_dir'] = video_dir + '%s/' % run_dir
        run_params['max_w_p'] = 1000 * run_params['a']  # pixels
        run_params['im_dt'] = 5  # time interval of each image
        run_params['t_switch'] = np.array([0, 155, 310, 425, 610, 825])  # instances when pump speed changes
        run_params['t_eq'] = run_params['t_switch'][1:] - 5
        run_params['pump_set'] = np.array([350, 250, 200, 150, 125])
        run_params['pump_flux'] = get_vol_flux(run_params['pump_set'])
        run_params['x_cut_off_cm'] = 68
        run_params['ds'] = 33*(3/4)
        run_params['tilt_fix'] = -0.6

    return run_params



def convert_mov2mp4(run_name, crop=True):
    # test cropping at command line, example:
    # ffmpeg -ss 0 -i july25_run/DSCN4556.MOV -vframes 1 -q:v 2 -filter:v crop=1180:360:200:100 test_images/crop_test.jpg

    # TODO: move to run params

    if run_name == 'july14_run1':
        input_video_path = video_dir + 'july14_run/DSCN4415.MOV'
        output_video_path = video_dir + 'july14_run/%s_cropped.mp4' % run_name
        crop_filter = "crop=1180:360:0:0"

    elif run_name == 'july17_run1':
        input_video_path = video_dir + 'july17_run/DSCN4425.MOV'
        output_video_path = video_dir + 'july17_run/%s_cropped.mp4' % run_name
        crop_filter = "crop=1180:360:240:120 "

    elif run_name == 'july20_run1':
        input_video_path = video_dir + 'july20_run/DSCN4468.MOV'
        output_video_path = video_dir + 'july20_run/%s_cropped.mp4' % run_name
        crop_filter = "crop=1180:360:240:120 "

    elif run_name == 'july21_run1':
        input_video_path = video_dir + 'july21_run/DSCN4489.MOV'
        output_video_path = video_dir + 'july21_run/%s_cropped.mp4' % run_name
        crop_filter = "crop=1180:360:240:120 "

    elif run_name == 'july24_run1':
        input_video_path = video_dir + 'july24_run/DSCN4541.MOV'
        output_video_path = video_dir + 'july24_run/%s_cropped.mp4' % run_name
        crop_filter = "crop=1180:360:240:120 "

    elif run_name == 'july25_run1':
        input_video_path = video_dir + 'july25_run/DSCN4556.MOV'
        output_video_path = video_dir + 'july25_run/%s_cropped.mp4' % run_name
        crop_filter = "crop=1180:360:240:120 "

    else:

        print("ERROR: run name not found.")
        return

    print("Running this function will delete any exisiting video called '%s'." %output_video_path)
    print("Enter 'c' to proceed, 'q' to exit")
    debug_here()

    if crop:
        ff = FFmpeg(inputs={input_video_path: None}, outputs={output_video_path: ['-filter:v', crop_filter, '-y']})
    else:
        ff = FFmpeg(inputs={input_video_path: None}, outputs={output_video_path: ['-y']})

    ff.run()


def add_pump_spd_text(run_name):
    if run_name == 'july12':
        input_video_path = video_dir + "july12_run/july12_run1_scaled.mp4"
        output_video_path = video_dir + "july12_run/july12_run1_scaled_with_text.mp4"
        text_times = [0, 60 * 1 + 35, 60 * 3 + 15, 60 * 3 + 45, 60 * 4 + 12, 60 * 4 + 59, 60 * 6 + 12, 7 * 60 + 44,
                      10 * 60 + 55, 14 * 60 + 14, 14 * 60 + 48]
        pump_speed = [250, 350, 325, 300, 250, 200, 150, 100, 200, 300]
        x_loc, y_loc = 10, 40

    elif run_name == 'july14':

        input_video_path = video_dir + "july14_run/july14_run1_cropped.mp4"
        output_video_path = video_dir + "july14_run/july14_run1_cropped_with_text.mp4"
        text_times = [0, 60 * 2 + 59, 60 * 4 + 22, 60 * 6 + 17, 60 * 8 + 3, 60 * 14 + 17]
        pump_speed = [350, 300, 250, 200, 150]
        x_loc, y_loc = 10, 40

    # fps = 23.98

    drawtext_str_list = []

    for i, spd in enumerate(pump_speed):
        t_start, t_end = text_times[i], text_times[i + 1] - 3
        drawtext_str_list.append("drawtext=fontfile=/Library/Fonts/Arial.ttf:text='%%{pts\:gmtime\\:0\\:%%M %%S}\rPump "
                                 "speed\: %s':fontcolor=white: "
                                 "fontsize=30: box=1:boxcolor=black@0.5:boxborderw=5:x=%s:y=%s:enable="
                                 "'between(t,%s,%s)'" % (spd, x_loc, y_loc, t_start, t_end))
    drawtext_str = ",".join(drawtext_str_list)
    ff = FFmpeg(inputs={input_video_path: None}, outputs={output_video_path: ['-vf', drawtext_str, '-y']})
    ff.run()


def speed_up_video(input_video_path, rate=10):
    # TODO: add run_name control via input argument

    input_fpath_split = input_video_path.split('/')
    input_dir = "".join(input_fpath_split[:-1])
    input_file = input_fpath_split[-1]
    file_root = input_file.split('.')[0]
    ext = input_file.split('.')[-1]
    output_video_path = os.path.join(input_dir, "%s_x%s.%s" % (file_root, rate, ext))
    # debug_here()
    # input_video_path = "july12_run/july12_run1_scaled_with_text.mp4"
    # output_video_path = "july12_run/july12_run1_scaled_with_text_x%s.mp4" % rate

    ff = FFmpeg(inputs={input_video_path: None},
                outputs={output_video_path: ['-filter:v', "setpts=%s*PTS" % (1 / rate), '-y']})
    ff.run()


def export_frames(run_name, fps=1):
    """Function to export frames from video as png images"""


    if run_name not in run_name_list:
        print("Error: run name not found.")
        return

    run_dir = run_name[:-1]
    input_video_path = video_dir + '%s/%s_cropped.mp4' %(run_dir, run_name)
    output_dir = video_dir + '%s/frames/' %run_dir

    output_video_path = os.path.join(output_dir, "image_%04d.png")

    ff = FFmpeg(inputs={input_video_path: None},
                outputs={output_video_path: ['-vf', 'fps=%s' % fps, "-qscale:v", "2", '-y']})
    ff.run()


def rotate_resize_images(run_name):

    import PIL
    from PIL import Image

    if run_name not in run_name_list:
        print("Error: run name not found.")
        return

    plt.close('all')
    x_scale = 2  # factor by which to increase x-resolution (number of pixels)
    y_scale = 2  # factor by which to increase y-resolution (number of pixels)
    t_save = 5  # seconds


    run_dir = run_name[:-1]
    frames_dir = video_dir + '%s/frames/' %run_dir
    output_dir = video_dir + "%s/rotated_rz_frames/" %run_dir

    # load run parameters
    run_params = get_image_settings(run_name)
    rot_angle = run_params["tilt_fix"]

    files = os.listdir(frames_dir)
    img_files = [f for f in files if f.endswith('.png')]

    print("Processing %s image files..." % len(img_files))

    for i, img_file in enumerate(img_files):

        # save every 5 images
        if (i + 1) % t_save == 0 or i == 0:
            img = Image.open(os.path.join(frames_dir, img_file))
            img2 = img.rotate(rot_angle, resample=Image.BILINEAR)

            width, height = img2.size
            img3 = img2.resize((width * x_scale, height * y_scale), PIL.Image.LANCZOS)

            img3.save(os.path.join(output_dir, img_file), compress_level=6)

            # debug_here()


def read_image_frames_test():
    import matplotlib.image as mpimg
    # import cv2

    plt.close('all')
    test_image1 = 'july12_run/frames/image_0160.png'
    test_image2 = 'july12_run/frames/image_0369.png'
    full_img1 = mpimg.imread(test_image1)
    full_img2 = mpimg.imread(test_image2)

    # plot full image
    plt.figure(figsize=(9, 6))
    plt.subplot(211)
    plt.imshow(full_img1)
    plt.title("Full frame: %s" % test_image1.rsplit('/')[-1])

    plt.subplot(212)
    plt.imshow(full_img2)
    plt.title("Full frame: %s" % test_image2.rsplit('/')[-1])

    # crop to include only tube
    tube_img1 = full_img1[260:340, :1100, :]
    tube_img2 = full_img2[260:340, :1100, :]
    rgb = 0
    asp = 2

    plt.figure(figsize=(10, 8))
    # plt.subplot(211)
    # plt.imshow(tube_img, aspect=asp)
    # plt.title("Image: %s, Cropped frame" %test_image1)

    plt.subplot(311)
    plt.imshow(tube_img1[:, :, rgb], aspect=asp, cmap='gray')
    plt.title("%s, Cropped frame, RGB channel = %s" % (test_image1.rsplit('/')[-1], rgb))

    plt.subplot(312)
    plt.imshow(tube_img2[:, :, rgb], aspect=asp, cmap='gray')
    plt.title("%s, Cropped frame, RGB channel = %s" % (test_image2.rsplit('/')[-1], rgb))

    plt.subplot(313)
    plt.imshow(tube_img2[:, :, rgb] - tube_img1[:, :, rgb], aspect=asp, cmap='gray')
    plt.title("image2 - image 1")

    plt.show()

    # an attempt to identify lines

    # # get edges of image 1
    # im1_edges = cv2.Canny(tube_img1[:, :, rgb], 100, 200, apertureSize=3)
    #
    # plt.figure()
    # plt.subplot(211)
    # plt.imshow(tube_img1, aspect=asp, cmap='gray')
    # plt.show()
    #
    # plt.subplot(212)
    # plt.imshow(im1_edges, aspect=asp, cmap='gray')
    # plt.show()


def line_detection1(im_num=650):
    """an attempt to identify lines"""

    import cv2

    # controls
    asp = 2

    plt.close('all')
    # test_image = 'july12_run/frames/image_0160.png'
    test_image1 = 'july14_run/frames/image_0001.png'
    test_image2 = 'july14_run/frames/image_%04d.png' % im_num
    img_name = test_image2.rsplit('/')[-1].split('.')[0]
    img1 = cv2.imread(test_image1)
    img2 = cv2.imread(test_image2)
    # img = img2 - img1

    # get red channel image (cv2 is BGR)
    # red_img1 = img1[:, :, 2]
    red_img2 = img2[:, :, 2]
    # red_img = red_img2 - red_img1
    # red_img[red_img<20] = 0
    # red_img[red_img > 50] = red_img.max()

    # get gray-scale image
    # gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gray_img = gray_img2 - gray_img1

    # crop images
    x_r = slice(100, 1200)
    y_r = slice(200, 330)
    img_cropped = img2[y_r, x_r, :]
    gray_img_cropped = gray_img2[y_r, x_r]
    red_img_cropped = red_img2[y_r, x_r]

    cv2.imwrite('%s.jpg' % img_name, img_cropped)

    # get edges
    # all_edges = cv2.Canny(gray_img, 50, 300, apertureSize=3)
    gray_edges = cv2.Canny(gray_img_cropped, 100, 300, apertureSize=3)
    red_edges = cv2.Canny(red_img_cropped, 100, 300, apertureSize=3)

    # # plot gray edges
    # plt.figure()
    # plt.subplot(211)
    # plt.imshow(gray_img1, aspect=asp, cmap='gray')
    # plt.show()
    #
    # plt.subplot(212)
    # plt.imshow(gray_img2, aspect=asp, cmap='gray')
    # plt.show()

    # plot gray edges
    plt.figure()
    plt.subplot(211)
    plt.imshow(gray_img_cropped, aspect=asp, cmap='gray')
    plt.show()

    plt.subplot(212)
    plt.imshow(gray_edges, aspect=asp, cmap='gray')
    plt.show()

    # plot red edges
    plt.figure()
    plt.subplot(211)
    plt.imshow(red_img_cropped, aspect=asp, cmap='gray')
    plt.show()

    plt.subplot(212)
    plt.imshow(red_edges, aspect=asp, cmap='gray')
    plt.show()

    # apply hough algorithm to  image
    theta_r = np.pi / 180  # angle precision
    len_r = 1  # length precision
    threshold = 200  # ??
    h_lines = cv2.HoughLines(gray_edges, len_r, theta_r, threshold)
    h_lines = np.squeeze(h_lines)
    # for x1, y1, x2, y2 in h_lines[0]:
    #     cv2.line(img_cropped, (x1, y1), (x2, y2), (0, 0, 255), 3)




    # x = (rho - y * sin( theta))/cos( theta)
    # base is roughly at y = 95
    tube_h_pix = 37  # pixels
    tube_x0_pix = 10  # pixels
    tube_y0_pix = 92  # pixels
    tube_h_cm = 2.5  # cm
    cm_per_pix = tube_h_cm / tube_h_pix

    # find where all lines cross tube base and top
    theta = h_lines[:, 1]
    rho = h_lines[:, 0]
    base_x_pix = np.floor((rho - tube_y0_pix * np.sin(theta)) / np.cos(theta))
    top_x_pix = np.floor((rho - (tube_y0_pix - tube_h_pix) * np.sin(theta)) / np.cos(theta))
    w_x_pix = base_x_pix - tube_x0_pix
    w_x_cm = w_x_pix * cm_per_pix

    # find lines with widths less than 80 cm and top intercepts less than tube_x0_pix
    wi = np.logical_and(np.abs(w_x_cm) < 100, top_x_pix < tube_x0_pix)

    if len(w_x_cm[wi]) >= 1:
        wedge_x_pix_mean = np.median(w_x_pix[wi])
        wedge_x_cm_mean = np.median(w_x_cm[wi])
        wedge_x_cm_std = np.std(w_x_cm[wi])

        print(w_x_cm[wi])

        print("Wedge width = %.2f cm +/- %.2f cm" % (wedge_x_cm_mean, wedge_x_cm_std))

    else:

        print("No wedge detected.")

    # debug_here()


    # get wedge width
    max_idx = np.argmax(np.rad2deg(theta) - 90)
    max_rho = h_lines[max_idx, 0]
    max_theta = h_lines[max_idx, 1]
    wedge_x_pix = int((max_rho - tube_y0_pix * np.sin(max_theta)) / np.cos(max_theta))
    wedge_x_cm = (wedge_x_pix - tube_x0_pix) * cm_per_pix
    print("Wedge width = %s pixels = %.2f cm" % (wedge_x_pix, wedge_x_cm))

    # debug_here()
    # draw lines on image
    num_lines, _ = h_lines.shape
    for i in range(num_lines):

        theta = h_lines[i, 1]
        theta_deg = np.rad2deg(theta) - 90
        rho = h_lines[i, 0]
        aa = np.cos(theta)
        bb = np.sin(theta)
        x0 = aa * rho
        y0 = bb * rho
        scale = 1000
        x1 = int(x0 + scale * -bb)
        y1 = int(y0 + scale * aa)
        x2 = int(x0 - scale * -bb)
        y2 = int(y0 - scale * aa)

        # x1 = h_lines[0][0]
        # x2 = h_lines[0][1]
        # y1 = h_lines[0][2]
        # y2 = h_lines[0][3]

        print(theta_deg)
        # debug_here()

        col = (0, 0, 0)
        lw = 1
        cv2.line(img_cropped, (x1, y1), (x2, y2), col, lw)

        if i == max_idx:
            # col = (50, 50, 150)
            col = (255, 0, 255)
            # col = (255, 255, 255)
            lw = 1
            cv2.line(img_cropped, (tube_x0_pix, tube_y0_pix - tube_h_pix), (tube_x0_pix + wedge_x_pix, tube_y0_pix),
                     col, lw)
            # else:
            #     col = (0, 0, 0)
            #     lw = 1
            #     cv2.line(gray_img_cropped, (x1, y1), (x2, y2), col, lw)

    # plot tube height
    cv2.line(img_cropped, (500, tube_y0_pix), (500, tube_y0_pix - tube_h_pix), (0, 255, 255), 1)

    # plot wedge width
    cv2.line(img_cropped, (tube_x0_pix, tube_y0_pix + 5), (tube_x0_pix + wedge_x_pix, tube_y0_pix + 5), (100, 20, 255),
             2)
    cv2.line(img_cropped, (tube_x0_pix, tube_y0_pix + 10), (tube_x0_pix + int(wedge_x_pix_mean), tube_y0_pix + 10),
             (255, 0, 255), 2)

    # plot tube baseline
    cv2.line(img_cropped, (0, tube_y0_pix), (1000, tube_y0_pix), (255, 255, 255), 1)

    cv2.imwrite('%s_wedge_width.jpg' % img_name, img_cropped)


def line_detection1_loop():
    import cv2

    frames_dir = 'july14_run/frames/'
    files = os.listdir(frames_dir)
    img_files = [f for f in files if f.endswith('.png')]

    # define tube parameters
    tube_h_pix = 37  # pixels
    tube_x0_pix = 10  # pixels
    tube_y0_pix = 92  # pixels
    tube_h_cm = 2.5  # cm
    cm_per_pix = tube_h_cm / tube_h_pix

    w_x_list = []
    print("Processing %s image files" % len(img_files))

    # set default length to be nans
    wedge_x_cm_mean = np.nan
    wedge_x_cm_std = np.nan

    for img_file in img_files:

        # read in image
        img = cv2.imread(os.path.join(frames_dir, img_file))

        # convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # crop image
        x_r = slice(100, 1200)
        y_r = slice(200, 330)
        # img_cropped = img[y_r, x_r, :]
        gray_img_cropped = gray_img[y_r, x_r]

        # get edges of grayscaled image
        gray_edges = cv2.Canny(gray_img_cropped, 100, 300, apertureSize=3)

        # apply hough algorithm to find lines
        theta_r = np.pi / 180  # angle precision
        len_r = 1  # length precision
        threshold = 200  # minimum line length
        h_lines = cv2.HoughLines(gray_edges, len_r, theta_r, threshold)
        h_lines = np.atleast_2d(np.squeeze(h_lines))

        if h_lines.shape[0] < 2:
            w_x_list.append([wedge_x_cm_mean, wedge_x_cm_std])
            continue

        # find where all lines cross tube base and top
        theta = h_lines[:, 1]
        rho = h_lines[:, 0]
        base_x_pix = np.floor((rho - tube_y0_pix * np.sin(theta)) / np.cos(theta))
        top_x_pix = np.floor((rho - (tube_y0_pix - tube_h_pix) * np.sin(theta)) / np.cos(theta))
        w_x_pix = base_x_pix - tube_x0_pix
        w_x_cm = w_x_pix * cm_per_pix

        # find lines with widths less than 80 cm and top intercepts less than tube_x0_pix
        wi = np.logical_and(np.abs(w_x_cm) < 100, top_x_pix < tube_x0_pix)

        if len(w_x_cm[wi]) >= 1:
            # wedge_x_pix_mean = np.median(w_x_pix[wi])

            if np.std(w_x_cm[wi]) > 10:
                wedge_x_cm_mean = np.median(w_x_cm[wi])
                wedge_x_cm_std = np.std(w_x_cm[wi])

        # print("Wedge width = %.2f cm +/- %.2f cm" % (wedge_x_cm_mean, wedge_x_cm_std))

        w_x_list.append([wedge_x_cm_mean, wedge_x_cm_std])

    # debug_here()

    w_x_arr = np.array(w_x_list)
    plt.figure()
    plt.errorbar(range(len(w_x_arr)), w_x_arr[:, 0], w_x_arr[:, 1], color='0.5')
    plt.plot(range(len(w_x_arr)), w_x_arr[:, 0], color='r')
    plt.grid(True)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Wedge Length (cm)")

    plt.show()



def wedge_detection_v2(run_name, output, im_test_num=250, plot_test=True):
    import cv2

    # load run parameters
    run_params = get_image_settings(run_name)

    # controls
    asp = 2  # aspect ratio of plots
    plot_dir = os.path.join(run_params['run_dir'], 'analysis_plots/')
    run_name_str = " ".join(run_name.split('_'))

    plt.close('all')

    files = os.listdir(run_params['frames_dir'])
    img_files = [f for f in files if f.endswith('.png')]

    # define tube parameters
    tube_h_cm = 2.5  # cm
    cm_per_pix = tube_h_cm / run_params['tube_h_p']

    w_x_list = []
    print("Processing %s image files" % len(img_files))

    for img_file in img_files:

        # read in image
        img = cv2.imread(os.path.join(run_params['frames_dir'], img_file))

        img_cropped = img[run_params['y_r'], run_params['x_r'], :]

        # convert to grayscale
        gray_img = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

        # get red channel intensity
        red_img = img_cropped[:, :, 2]
        blue_img = img_cropped[:, :, 0]

        # normalize
        red_img = red_img / np.max(red_img)
        blue_img = blue_img / np.max(blue_img)
        gray_img = gray_img / np.max(gray_img)

        # plot pixel brightness along line inside tube
        y_i = run_params['tube_y0_p'] + run_params['base_offset_pix']
        gc_tube = gray_img[y_i, :]
        rc_tube = red_img[y_i, :]
        bc_tube = blue_img[y_i, :]

        # find where the first time red channel intensity drops below
        # threshold for 10 consecutive pixels
        ri_idx = []
        for idx, ri in enumerate(rc_tube):

            if ri < run_params['ri_thresh']:
                # color indices where ri < threshold
                ri_idx.append(idx)
            else:
                # reset ri_idx
                ri_idx = []

            if len(ri_idx) > run_params['ri_thresh_minL'] and ri_idx[0]>run_params['tube_x0_p']:
                break

        if len(ri_idx) == 0:
            w_x_list.append(np.nan)
        else:

            # use first index of ri_idx as start of wedge
            wedge_x_pix = ri_idx[0] - run_params['tube_x0_p']
            wedge_x_cm = wedge_x_pix * cm_per_pix

            if 0 < wedge_x_cm <= 100:
                w_x_list.append(wedge_x_cm)
            else:
                w_x_list.append(np.nan)

        if img_file == 'image_%04d.png' % im_test_num and plot_test:
            img_root = img_file.split('.')[0]

            # plot cropped images
            fig = plt.figure(figsize=(8, 9))
            plt.subplot(311)
            im = plt.imshow(gray_img, aspect=asp, cmap='gray')
            plt.hlines(y_i, 0 * run_params['a'], 1200 * run_params['a'], color='b')
            plt.hlines(run_params['tube_y0_p'], 0 * run_params['a'], 1200 * run_params['a'], color='b')
            plt.title("gray scale")
            ax = plt.gca()
            ax.spines['bottom'].set_color('red')
            ax.xaxis.label.set_color('red')
            ax.tick_params(axis='x', colors='red')
            plt.grid(True, color='r')

            plt.subplot(312)
            plt.imshow(red_img, aspect=asp, cmap='gray')
            plt.hlines(y_i, 0 * run_params['a'], 1200 * run_params['a'], color='b')
            plt.hlines(run_params['tube_y0_p'], 0 * run_params['a'], 1200 * run_params['a'], color='b')
            plt.title("red channel")
            ax = plt.gca()
            ax.spines['bottom'].set_color('red')
            ax.xaxis.label.set_color('red')
            ax.tick_params(axis='x', colors='red')
            plt.grid(True, color='r')

            plt.subplot(313)
            plt.imshow(blue_img, aspect=asp, cmap='gray')
            plt.hlines(y_i, 0 * run_params['a'], 1200 * run_params['a'], color='b')
            plt.hlines(run_params['tube_y0_p'], 0 * run_params['a'], 1200 * run_params['a'], color='b')
            plt.title("blue channel")
            ax = plt.gca()
            ax.spines['bottom'].set_color('red')
            ax.xaxis.label.set_color('red')
            ax.tick_params(axis='x', colors='red')
            plt.grid(True, color='r')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            plt.savefig("%s%s_color_intensity.png" % (plot_dir, img_root), dpi=300, bbox_inches='tight')

            # plot intensity
            plt.figure()
            plt.title("%s test case" % img_file)
            plt.plot(gc_tube, color='0.5', label="gray scale")
            plt.plot(rc_tube, color='r', label="red")
            plt.plot(bc_tube, color='b', label="blue")
            plt.ylim(0, 1.2)
            plt.legend()
            plt.grid(True)

            plt.savefig("%s%s_tube_brightness.pdf" % (plot_dir, img_root), bbox_inches='tight')

            plt.show()

            debug_here()

    # plot wedge length versus time
    w_x_arr = np.array(w_x_list)
    plt.figure()
    # plt.errorbar(range(len(w_x_arr)), w_x_arr[:, 0], w_x_arr[:, 1], color='0.5')
    time = range(0, run_params['im_dt'] * len(w_x_arr), run_params['im_dt'])
    plt.plot(time, w_x_arr, color='r')
    plt.grid(True)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Wedge Length (cm)")
    plt.ylim(0, 80)
    plt.title("Wedge length versus time (%s)" % run_name_str)

    # add vertical lines denoting pump range
    plt.vlines(run_params['t_switch'], 0, 80, linestyles='--', colors='0.6', linewidth=2)
    # for i, pump_set in enumerate(run_params['pump_set']):
    #     x_pos = run_params['t_switch'][i] + (run_params['t_switch'][i+1] - run_params['t_switch'][i])/2
    #     vol_flux = get_vol_flux(pump_set)
    #     # plt.text(x_pos, run_params['pump_set_text_ypos'], "Q = %.1f cm$^3$" %vol_flux, fontsize=10)

    plt.savefig("%swedge_length_time.pdf" % plot_dir, bbox_inches='tight')

    # plot equilibrium wedge length versus pump speed
    w_len_eq = []
    for t_eq in run_params['t_eq']:
        t_eq_idx = np.argmin(np.abs(time - t_eq))
        w_len_eq.append(w_x_arr[t_eq_idx])

    w_len_eq = np.array(w_len_eq)

    plt.figure()
    plt.plot(run_params['pump_flux'], w_len_eq, '-o', markersize=10, linewidth=2)
    plt.ylabel("Equilibrium Wedge length (cm)")
    plt.xlabel("Volume flux (cm$^3$/s)")
    plt.title("Equilibrium wedge length versus pump speed (%s)" % run_name_str)
    plt.grid(True)

    plt.savefig("%seq_wedge_length_pspd.pdf" % plot_dir, bbox_inches='tight')

    plt.show()

    # wedge_length_stats = {"time": time, "w_len": w_x_arr, "w_len_eq": w_len_eq, "ds": run_params["ds"]}

    output[run_name] = {"time": time, "w_len": w_x_arr, "w_len_eq": w_len_eq, "ds": run_params["ds"]}

    return output


def wedge_detection_v3(run_name, ds):
    import cv2

    plt.close('all')

    run_name_str = " ".join(run_name.split('_'))

    run_params = get_image_settings(run_name)
    plot_dir = os.path.join(run_params['run_dir'], 'analysis_plots/')

    files = os.listdir(run_params['frames_dir'])
    img_files = [f for f in files if f.endswith('.png')]

    # define tube parameters
    tube_h_cm = 2.5  # cm
    cm_per_pix = tube_h_cm / run_params['tube_h_p']

    print("Processing %s image files" % len(img_files))

    wedge_h_x = []
    tstep = []
    for img_file in img_files:

        # get time step from file name

        tstep.append(int(img_file.split('.')[0].split('_')[-1]))

        # read in image
        img = cv2.imread(os.path.join(run_params['frames_dir'], img_file))

        img_cropped = img[run_params['y_r'], run_params['x_r'], :]

        # convert to grayscale
        # gray_img = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

        # get red channel intensity
        red_img = img_cropped[:, :, 2]
        # blue_img = img_cropped[:, :, 0]

        # normalize
        red_img = red_img / np.max(red_img)
        # blue_img = blue_img / np.max(blue_img)
        # gray_img = gray_img / np.max(gray_img)

        # r_thresh = 0.02
        wedge_h_xi = []
        for i in range(run_params['tube_x0_p'], run_params['tube_x0_p'] + run_params['max_w_p']):

            # take vert section from base of tube to top of tube
            y0 = run_params['tube_y0_p']
            y_sect = red_img[y0 - run_params['tube_h_p']:y0, i]

            # reverse y_sect so it is ordered from top to bottom
            y_sect_r = y_sect[::-1]

            # loop through vert section to find where freshwater layer starts
            # find where the first time red channel intensity drops below
            # threshold for 10 consecutive pixels
            ri_idx = []
            for j, ri in enumerate(y_sect_r):

                if ri < run_params['ri_thresh']:
                    # color indices where ri < threshold
                    ri_idx.append(j)
                else:
                    # reset ri_idx
                    ri_idx = []

                if len(ri_idx) > run_params['ri_thresh_minL'] / 2:
                    break

            if len(ri_idx) < run_params['ri_thresh_minL'] / 2:
                wedge_h_xi.append(np.nan)
            else:
                # use first index of ri_idx as start of wedge
                wedge_x_cm = ri_idx[0] * cm_per_pix

                if wedge_x_cm > 0.2:
                    wedge_h_xi.append(wedge_x_cm)
                else:
                    wedge_h_xi.append(np.nan)

                    # debug_here()

        # debug_here()
        # save height profile for image
        wedge_h_x.append(wedge_h_xi)

    # debug_here()
    # get true wedge height in cm and mask potentially bad data
    wedge_h_x_arr = np.array(wedge_h_x)
    along_tube_distance = np.array(range(len(wedge_h_x_arr[0, :]))) * cm_per_pix

    x_mask = np.flatnonzero(along_tube_distance > run_params['x_cut_off_cm'])
    wedge_h_x_arr[:, x_mask] = np.nan
    wedge_h_x_arr_true = wedge_h_x_arr - np.abs(run_params['base_offset_pix']) * cm_per_pix  # true wedge height
    wedge_h_x_arr_true[wedge_h_x_arr_true<=0] = np.nan

    # save data
    wedge_evol = {"wedge_h_x": wedge_h_x_arr_true, "tube_x": along_tube_distance}
    pickle.dump(wedge_evol, open(os.path.join(run_params['run_dir'], "july17_run1_wedge_evol.p"), "wb"))

    # generate plots
    _, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes = axes.flatten()

    # define color cycle
    cmap1 = copy.copy(plt.get_cmap('rainbow'))
    cmap2 = copy.copy(plt.get_cmap('rainbow'))

    tsteps_full = np.array(tstep)
    ti = np.logical_and(tsteps_full>=run_params['t_switch'][1], tsteps_full<=run_params['t_switch'][-1])
    tsteps_plot = tsteps_full[ti]
    num_colors1 = len(tsteps_plot)
    colors1 = cmap1(np.linspace(0, 1, num_colors1))
    colors2 = cmap2(np.linspace(0, 1, len(run_params['t_switch'][1:])))
    ii = 0
    j = 0
    for i, t in enumerate(tsteps_plot):

        # if t < run_params['t_switch'][1] or t > run_params['t_switch'][-1]:
        #     continue

        # TODO: compute average equilbrium shape over finite time window

        # debug_here()
        idx = np.flatnonzero(t==tsteps_full)[0]
        w_h_cm_true = wedge_h_x_arr_true[idx, :]
        tube_w_cm = 2.5
        beta = 0.8e-3  # 1/psu
        # ds = 33  # psu (now comes from input)
        g = 9.81 * 100  # cm/s2
        h1 = 2.5 - w_h_cm_true
        Fr1 = run_params['pump_flux'][j] / np.sqrt(tube_w_cm ** 2 * h1 ** 3 * beta * ds * g)

        lw = 0.5
        alpha = 0.4
        # highlight wedge at equilbrium positions
        if t in run_params['t_switch'][1:]:  # TODO: Sort this out
            lw = 3
            alpha = 1
            axes[0].plot(along_tube_distance, w_h_cm_true, color=colors1[ii], lw=lw, alpha=alpha,
                         label="Q'=%s at t=%ss" % (run_params['pump_set'][j], t), zorder=10)

            axes[1].plot(along_tube_distance, Fr1, color=colors2[j], lw=lw, alpha=alpha,
                         label="Q'=%s at t=%ss" % (run_params['pump_set'][j], t), zorder=10)
            j = j + 1
        else:
            axes[0].plot(along_tube_distance, w_h_cm_true, color=colors1[ii], lw=lw, alpha=alpha)
            # axes[1].plot(along_tube_distance, Fr1, color=colors[i], lw=lw, alpha=alpha)

        ii = ii+1

    plt.sca(axes[0])
    plt.grid(True)
    plt.ylabel("Wedge height (cm)")
    plt.xlabel("Along tube distance (cm)")
    plt.ylim(0, 2.5)
    plt.xlim(0, 70)
    plt.title("Wedge evolution for %s" % run_name_str)
    plt.legend(loc=0, fontsize=8, ncol=2)

    plt.sca(axes[1])
    plt.grid(True)
    plt.ylabel("Local Froude Number")
    plt.xlabel("Along tube distance (cm)")
    plt.xlim(0, 70)
    plt.ylim(0, 1.5)
    plt.title("Layer 1 Froude number ($Q/\sqrt{g' h^3 w^2}$) for %s" % run_name_str)
    plt.legend(loc=0, fontsize=8, ncol=2)

    plt.subplots_adjust(hspace=0.4)
    plt.savefig("%swedge_evolution.pdf" % plot_dir, bbox_inches='tight')

    plt.show()


def get_vol_flux(pump_setting):
    """ Function that converts pump setting to volume flux (cm3/s)
        Pump: ISNATEC BVP=Z
        Pump head serial #: 884519
    """

    return pump_setting * 0.05 - 0.78


def run_comparison(lab_runs):


    plot_dir = os.path.join(video_dir, 'analysis/')

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes = axes.flatten()

    for run_name in run_name_list:

        run_params = get_image_settings(run_name)

        if run_name in ["july17_run1", "july14_run1", "july24_run1", "july25_run1"]:
            ls = '-o'
        else:
            ls = 'o'

        # plot sorted pump speeds
        lab_run = lab_runs[run_name]
        pset_sort_i = np.argsort(run_params['pump_set'])
        pump_setting =run_params['pump_set'][pset_sort_i]
        pump_flux = run_params['pump_flux'][pset_sort_i]
        w_len_eq = lab_run["w_len_eq"][pset_sort_i]


        # axes[0].plot(lab_run["time"], lab_run["w_len"], label=lab_run['run_name'])
        axes[0].plot(pump_setting, w_len_eq, ls, markersize=8, linewidth=2,
                     label="%s (ds=%s)" % (run_name, lab_run["ds"]), alpha=0.5)
        axes[1].plot(pump_flux, w_len_eq, ls, markersize=8, linewidth=2,
                     label="%s (ds=%s)" %(run_name, lab_run["ds"]), alpha=0.5)


        # get pump flux scaled by reduced gravity (i.e. froude number)
        g = 9.81 * 100  # cm/s2
        beta = 0.8e-3  # haline contraction co-eff (1/psu)
        h0 = 2.5  # cm
        w0 = 2.5  # cm
        Fr0 = pump_flux/np.sqrt(g * beta * lab_run['ds'] * w0**2 * h0**3)
        axes[2].plot(Fr0, w_len_eq, ls, markersize=8, linewidth=2,
                     label="%s (ds=%s)" % (run_name, lab_run["ds"]), alpha=0.5)

    plt.sca(axes[0])
    plt.ylabel("Equilibrium Wedge length (cm)")
    plt.xlabel("Pump Setting")
    plt.legend(loc=0, fontsize=8, ncol=2)
    plt.grid(True)
    plt.xlim(50, 450)

    plt.sca(axes[1])
    plt.ylabel("Equilibrium Wedge length (cm)")
    plt.xlabel("Volume flux $Q$ (cm$^3$/s)")
    # plt.legend(loc=0, fontsize=10)
    plt.grid(True)
    plt.xlim(0, 20)

    # ax = plt.gca()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12, ncol=1)

    plt.sca(axes[2])
    plt.ylabel("Equilibrium Wedge length (cm)")
    plt.xlabel("Full Channel Froude Number ($Q/\sqrt{g' H^3 W^3}$)")
    # plt.legend(loc=0, fontsize=10)
    plt.grid(True)
    # plt.xlim(0, 1)

    plt.subplots_adjust(hspace=0.4)
    plt.savefig("%srun_comparisons2.pdf" %plot_dir, bbox_inches="tight")

# main script
# add_pump_spd_text()
# speed_up_video(10)
