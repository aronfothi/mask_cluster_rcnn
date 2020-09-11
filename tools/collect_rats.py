
import sys
import os
import cv2
import numpy as np
import skimage.measure
import math


PATH_VIDEO_INPUT = '/media/hdd/aron/rats/Exp_2019_03_22_retrograd_study_20190415_160817.avi'
PATH_VIDEO_A_OUTPUT = None #'/home/vavsaai/databases/rats/rat_sepbg_a.avi'  # str or None
PATH_VIDEO_E_OUTPUT = None #'/home/vavsaai/databases/rats/rat_sepbg_e.avi'  # str or None
PATH_VIDEO_COMPOSITE_OUTPUT = '/media/hdd/aron/rats/rat_sepbg.avi'
GENERATE_FRAMES_FOLDER = '/media/hdd/aron/rats/frames_sep/'  # str or None
FR_START_IDX = 20
FR_END_IDX_EXCL = 30000
CUT_BBOX_YXHW = (160, 320, 420, 640)  # tuple(y, x, h, w)

# good background sampling frame intervals in ..._retrograd_study_20190415_160817: (0,500), (10580,10760)

# Algorithm is based on https://forums.fast.ai/t/part-3-background-removal-with-robust-pca/4286

COMPUTE_BACKGROUND_ONCE = True   # if True, bg of the first slice is used for all further slices
COMPUTE_SLICE_LEN = 300

BRIGHTNESS_THRES = 64   # default: 64
APPLY_ERODE_INTENSITY = 0  # set to <= 0 to disable


def _fast_mode_bin_3d(xs, n_bins):
    '''
    Compute multiple 3D histograms at once (up to several magnitudes faster than np.histogramdd() called in loop).
    Parameters:
        xs: ndarray(n_parallel, n_xs, 3) of uint
        n_bins: int
    Returns:
        mode_bin_idxs: ndarray(n_parallel, 3) of int32
    '''
    assert xs.shape[2:] == (3,)
    assert n_bins < 64
    bin_size = int(256/n_bins)
    n_flatbins = n_bins*n_bins*n_bins
    n_parallel = xs.shape[0]
    xs_bin_idxs = (xs // bin_size).astype(np.int32)   # (n_parallel, n_xs, 3) of int32
    xs_bin_flat_idxs = np.ravel_multi_index((xs_bin_idxs[:,:,0], xs_bin_idxs[:,:,1], xs_bin_idxs[:,:,2]),\
                                                    dims=(n_bins, n_bins, n_bins))  # (n_parallel, n_xs) of int32
    xs_par_idxs = np.arange(n_parallel, dtype=np.int32)  # (n_parallel,) of int32
    xs_par_idxs = np.broadcast_to(xs_par_idxs[:,None], shape=xs_bin_flat_idxs.shape)  # (n_parallel, n_xs) of int32
    xs_bincounts = np.zeros((n_parallel, n_flatbins), dtype=np.int32)  # (n_parallel, n_flatbins)
    np.add.at(xs_bincounts, (xs_par_idxs, xs_bin_flat_idxs), 1)  
                                            # xs_bincounts[(xs_par_idxs, xs_bin_flat_idxs)] += 1, unbuffered
    xs_modebin_flat_idxs = np.argmax(xs_bincounts, axis=1)    # (n_parallel,) of int32, flatbin idxs
    xs_modebin_idxtup = np.unravel_index(xs_modebin_flat_idxs, shape=(n_bins, n_bins, n_bins))
    mode_bin_idxs = np.stack(xs_modebin_idxtup, axis=-1)
    return mode_bin_idxs


def separate_mode(ims, im_A=None):
    '''
    Parameters:
        ims: ndarray(n_fr, sy, sx, n_ch) of uint8
        im_A: None OR ndarray(sy, sx, n_ch) of uint8; if given, mode is not calculated, im_A is used as the background
    Returns:
        im_A: ndarray(n_fr, sy, sx, n_ch) of uint8
        ims_E: ndarray(n_fr, sy, sx, n_ch) of uint8
    '''
    N_BINS = 32
    assert ims.dtype == np.uint8
    assert ims.shape[-1] == 3
    if im_A is None:
        orig_shape = ims.shape
        ims = ims.reshape((ims.shape[0], -1, ims.shape[-1]))  # (n_fr, sy*sx, n_ch) of uint8
        mode_bin_idxs = _fast_mode_bin_3d(ims.transpose((1,0,2)), N_BINS)   # (sy*sx, 3) of int32
        bin_size = int(256/N_BINS)
        bin_edges = np.append(np.arange(N_BINS)*bin_size, 255)   # (n_bins+1)
        bin_centers = (np.arange(N_BINS)*bin_size + (bin_size/2)).astype(np.float32)   # (n_bins)

        # fg: abs(mode_bin_center - pix)
        im_A_fl = bin_centers[mode_bin_idxs]
        im_A_fl = im_A_fl.reshape(orig_shape[1:])
        im_A = im_A_fl.astype(np.uint8, copy=False)
        ims = ims.reshape(orig_shape)
    else:
        assert im_A.shape == ims.shape[1:]
        assert im_A.dtype == np.uint8
        im_A_fl = im_A.astype(np.float32, copy=False)

    ims_E = np.fabs(im_A_fl - ims.astype(np.float32)).astype(np.uint8)
    return im_A, ims_E

def get_rat_regions(im_E, prev_reg_data, brightness_thres, n_objs, min_ratio_to_biggest):
    '''
    Searches for image regions in a brightness-thresholded image.
    Returns the centroid, bbox and mask of the top 'n_objs' biggest regions 
        which have a higher size ratio to the largest region than 'min_ratio_to_biggest'.
    Parameters:
        im_E: ndarray(sy, sx, n_ch) of uint8
        prev_reg_data: None OR same type as 'regions_data'
        brightness_thres: float; thresholding ims_E for pixels brighter than this limit
        n_objs: int
        min_ratio_to_biggest: float
    Returns:
        regions_data: list(n_big_objs_found) of 
                            tuple(cy, cx, bbox_miny, minx, maxy, maxx, mask:ndarray(bbox_h, bbox_h));
                                     n_big_objs_found <= n_objs;

    '''
    assert im_E.shape[2:] == (3,)
    assert 0. <= brightness_thres < 255.
    assert n_objs >= 1
    assert 0. <= min_ratio_to_biggest < 1.

    # threshold image
    im_E = np.amax(im_E, axis=-1)  # (sy, sx) of uint8
    im_E = (im_E > brightness_thres)  # (sy, sx) of bool_

    # erode mask optionally
    if APPLY_ERODE_INTENSITY > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (APPLY_ERODE_INTENSITY, APPLY_ERODE_INTENSITY))
        im_E = cv2.erode(im_E.astype(np.uint8), erode_kernel)

    # find connected foreground regions
    im_E, n_labels = skimage.measure.label(im_E, return_num=True)
    rprops = skimage.measure.regionprops(im_E, cache=True)  # list of regionprops

    # select top 'n_objs' biggest regions that are big enough
    biggest_reg_area = np.amax([rprop.area for rprop in rprops])
    min_obj_area = float(min_ratio_to_biggest)*biggest_reg_area
    biggest_reg_idxs = np.argsort([rprop.area for rprop in rprops])[-n_objs:][::-1]
    assert 1 <= biggest_reg_idxs.shape[0] <= n_objs

    # get cetroids for selected regions
    regions_data = []
    for reg_idx in biggest_reg_idxs:
        if rprops[reg_idx].area < min_obj_area:
            continue
        cy, cx = rprops[reg_idx].centroid
        miny, minx, maxy, maxx = rprops[reg_idx].bbox
        mask = rprops[reg_idx].image
        regions_data.append((cy, cx, miny, minx, maxy, maxx, mask))

    # tracking - only implemented for n_objs == 2 
    #   (switch order of objects in regions_data if mean obj distances show a swap in order compared to prev_reg_data)
    if (prev_reg_data is not None) and (len(prev_reg_data) == 2) and (len(regions_data) == 2):
        meandist_no_switch = (math.sqrt((prev_reg_data[0][0] - regions_data[0][0])**2. +\
                                (prev_reg_data[0][1] - regions_data[0][1])**2.) +\
                             math.sqrt((prev_reg_data[1][0] - regions_data[1][0])**2. +\
                                (prev_reg_data[1][1] - regions_data[1][1])**2.))*0.5
        meandist_switch = (math.sqrt((prev_reg_data[0][0] - regions_data[1][0])**2. +\
                                (prev_reg_data[0][1] - regions_data[1][1])**2.) +\
                           math.sqrt((prev_reg_data[1][0] - regions_data[0][0])**2. +\
                                (prev_reg_data[1][1] - regions_data[0][1])**2.))*0.5
        if meandist_switch < meandist_no_switch:
            regions_data = regions_data[::-1]

    return regions_data

def render_rat_regions(im_E, regions_data, colors_bgr8_tup, render_centroid=True, centroid_radius=5,\
                         render_mask=True, mask_alpha=0.5, render_info=True):
    '''
    Renders centroids and/or masks over image.
    Parameters:
        im_E: ndarray(sy, sx, n_ch) of uint8
        regions_data: <same as get_rat_regions() output format>
        colors_bgr8_tup: list of color tuples (bgr uint8) for each object
        render_centroid: bool
        centroid_radius: int
        render_mask: bool
        mask_alpha: float
        render_info: bool
    Returns:
        im_E: ndarray(sy, sx, n_ch) of uint8
    '''
    assert len(colors_bgr8_tup) >= len(regions_data)
    colors_arr = np.array(colors_bgr8_tup, dtype=np.uint8)
    assert colors_arr.shape[1:] == (3,)
    if render_mask:
        mask = np.zeros_like(im_E)
        for reg_idx in range(len(regions_data)):
            cy, cx, miny, minx, maxy, maxx, mask_in_bbox = regions_data[reg_idx]
            mask[miny:maxy, minx:maxx, :] += mask_in_bbox[:,:,None] * colors_arr[reg_idx]
        im_E = cv2.addWeighted(im_E, 1.-mask_alpha, mask, mask_alpha, 0.)

    if render_centroid:
        for reg_idx in range(len(regions_data)):
            cy, cx, miny, minx, maxy, maxx, mask_in_bbox = regions_data[reg_idx]
            color_tup = (int(colors_arr[reg_idx][0]), int(colors_arr[reg_idx][1]), int(colors_arr[reg_idx][2]))
            cv2.circle(im_E, (int(cx), int(cy)), radius=centroid_radius, color=color_tup, thickness=-1)

    if render_info:
        if len(regions_data) == 2:
            text = 'SEPARATED'
            text_color = (0,255,0)
        elif len(regions_data) == 1:
            text = 'ANNOTATION NEEDED'
            text_color = (64,64,192)
        elif len(regions_data) == 0:
            text = 'ANNOTATION NEEDED (NONE FOUND)'
            text_color = (0,0,255)
        cv2.putText(im_E, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, text_color, 2)

    return im_E


if __name__ == '__main__':

    assert (PATH_VIDEO_A_OUTPUT is not None) or (PATH_VIDEO_E_OUTPUT is not None) or\
                                                (PATH_VIDEO_COMPOSITE_OUTPUT is not None)

    if GENERATE_FRAMES_FOLDER is not None:
        folder_sep = os.path.join(GENERATE_FRAMES_FOLDER, 'sep')
        folder_joint = os.path.join(GENERATE_FRAMES_FOLDER, 'joint')
        os.makedirs(folder_sep, exist_ok=True)
        os.makedirs(folder_joint, exist_ok=True)

    assert FR_START_IDX < FR_END_IDX_EXCL
    RAT_COLORS_BGR = [(255,0,255), (255,255,0)]
    regions_data = None

    # video capture
    vid_capture = cv2.VideoCapture(PATH_VIDEO_INPUT)
    assert vid_capture.isOpened(), "Unable to open video file for reading: " + vid_in_path

    # video writers
    if PATH_VIDEO_A_OUTPUT is not None:
        # vr_fourcc = cv2.VideoWriter_fourcc(*'H264')     # use this codec with avi, Python2
        vr_fourcc = cv2.VideoWriter_fourcc(*'MJPG')     # use this codec with avi, Python3
        vr_fps = vid_capture.get(cv2.CAP_PROP_FPS)
        vr_frSize_xy = (CUT_BBOX_YXHW[3], CUT_BBOX_YXHW[2])
        vid_writer_a = cv2.VideoWriter(PATH_VIDEO_A_OUTPUT, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
        assert vid_writer_a.isOpened(), "Unable to open video file for writing: " + PATH_VIDEO_A_OUTPUT

    if PATH_VIDEO_E_OUTPUT is not None:
        # vr_fourcc = cv2.VideoWriter_fourcc(*'H264')     # use this codec with avi, Python2
        vr_fourcc = cv2.VideoWriter_fourcc(*'MJPG')     # use this codec with avi, Python3
        vr_fps = vid_capture.get(cv2.CAP_PROP_FPS)
        vr_frSize_xy = (CUT_BBOX_YXHW[3], CUT_BBOX_YXHW[2])
        vid_writer_e = cv2.VideoWriter(PATH_VIDEO_E_OUTPUT, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
        assert vid_writer_e.isOpened(), "Unable to open video file for writing: " + PATH_VIDEO_E_OUTPUT

    if PATH_VIDEO_COMPOSITE_OUTPUT is not None:
        # vr_fourcc = cv2.VideoWriter_fourcc(*'H264')     # use this codec with avi, Python2
        vr_fourcc = cv2.VideoWriter_fourcc(*'MJPG')     # use this codec with avi, Python3
        vr_fps = vid_capture.get(cv2.CAP_PROP_FPS)
        vr_frSize_xy = (2*CUT_BBOX_YXHW[3], 2*CUT_BBOX_YXHW[2])
        vid_writer = cv2.VideoWriter(PATH_VIDEO_COMPOSITE_OUTPUT, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
        assert vid_writer.isOpened(), "Unable to open video file for writing: " + PATH_VIDEO_COMPOSITE_OUTPUT


    #
    ims = []
    curr_bg = None
    fr_idx = 0
    while fr_idx < FR_END_IDX_EXCL:

        if fr_idx < FR_START_IDX:
            ret, curr_fr = vid_capture.read()
            fr_idx += 1
            continue

        ims = []
        print("    at frame idx#" + str(fr_idx))
        for batch_idx in range(COMPUTE_SLICE_LEN):
            ret, curr_fr = vid_capture.read()
            fr_idx += 1

            if (not ret) or (fr_idx >= FR_END_IDX_EXCL):
                break

            curr_fr = curr_fr[CUT_BBOX_YXHW[0]:CUT_BBOX_YXHW[0]+CUT_BBOX_YXHW[2], \
                              CUT_BBOX_YXHW[1]:CUT_BBOX_YXHW[1]+CUT_BBOX_YXHW[3],:]
            ims.append(curr_fr)

        # generate foreground (and background) images
        ims = np.stack(ims, axis=0)

        if (curr_bg is not None) and (COMPUTE_BACKGROUND_ONCE or (ims.shape[0] < COMPUTE_SLICE_LEN)):
            _, ims_E = separate_mode(ims, im_A=curr_bg)
        else:
            curr_bg, ims_E = separate_mode(ims)

        # find rats in foreground image
        if (PATH_VIDEO_COMPOSITE_OUTPUT is not None) or (GENERATE_FRAMES_FOLDER is not None):
            masked_ims = []
            simple_mask_ims = []
            rats_joint = []
            for im_E in ims_E:
                regions_data = get_rat_regions(im_E, regions_data, brightness_thres=BRIGHTNESS_THRES,\
                                              n_objs=2, min_ratio_to_biggest=0.3) # (n_objs_found, 2:[y,x])

                rats_joint.append(len(regions_data) < 2)
                masked_im = np.zeros_like(im_E)
                masked_im = render_rat_regions(masked_im, regions_data, colors_bgr8_tup=RAT_COLORS_BGR,\
                                         render_centroid=False, centroid_radius=5,\
                                         render_mask=True, mask_alpha=1., render_info=True)
                if GENERATE_FRAMES_FOLDER is not None:
                    simple_mask_im = np.zeros_like(im_E)
                    simple_mask_im = render_rat_regions(simple_mask_im, regions_data,\
                                             colors_bgr8_tup=[(255,255,255), (127,127,127)],\
                                             render_centroid=False, centroid_radius=5,\
                                             render_mask=True, mask_alpha=1., render_info=False)
                    simple_mask_im = np.amax(simple_mask_im, axis=-1)
                    simple_mask_ims.append(simple_mask_im)

                masked_ims.append(masked_im)

        # write images to video
        
        if PATH_VIDEO_A_OUTPUT is not None:
            for vidfr_idx in range(ims_E.shape[0]):
                vid_writer_a.write(curr_bg)
            
        if PATH_VIDEO_E_OUTPUT is not None:
            for vidfr_idx in range(ims_E.shape[0]):
                vid_writer_e.write(ims_E[vidfr_idx])

        if PATH_VIDEO_COMPOSITE_OUTPUT is not None:
            for vidfr_idx in range(ims.shape[0]):
                fig_top = np.concatenate([ims[vidfr_idx], curr_bg], axis=1)
                fig_bottom = np.concatenate([ims_E[vidfr_idx], masked_ims[vidfr_idx]], axis=1)
                fig = np.concatenate([fig_top, fig_bottom], axis=0)
                vid_writer.write(fig)

        if GENERATE_FRAMES_FOLDER is not None:
            fr_offset = fr_idx - ims.shape[0]
            for vidfr_idx in range(ims.shape[0]):
                folder_tag = 'joint' if rats_joint[vidfr_idx] else 'sep'
                im_fpath = os.path.join(GENERATE_FRAMES_FOLDER, folder_tag, 'im_' + str(fr_offset + vidfr_idx) + '.png')
                mask_fpath = os.path.join(GENERATE_FRAMES_FOLDER, folder_tag, 'mask_' + str(fr_offset + vidfr_idx) + '.png')
                cv2.imwrite(im_fpath, ims[vidfr_idx])
                cv2.imwrite(mask_fpath, simple_mask_ims[vidfr_idx])
                


    
    if PATH_VIDEO_A_OUTPUT is not None:
        vid_writer_a.release()
    if PATH_VIDEO_E_OUTPUT is not None:
        vid_writer_e.release()
    if PATH_VIDEO_COMPOSITE_OUTPUT is not None:
        vid_writer.release()
    
    vid_capture.release()


