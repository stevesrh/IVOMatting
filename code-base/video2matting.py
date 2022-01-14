import argparse
import os
import cv2
import numpy as np
import toml
from torch.nn import functional as F
import utils
from   utils import CONFIG
import networks
import torch

CODEC = "mp4v"
# SAVE_EXT = "mp4"

def single_inference(model, image_dict, post_process=False):
    with torch.no_grad():
        image, mask = image_dict['image'], image_dict['mask']
        alpha_shape = image_dict['alpha_shape']
        image = image.cuda()
        mask = mask.cuda()
        pred = model(image, mask)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1,
                                                        train_mode=False)
        alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2,
                                                        train_mode=False)
        alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

        h, w = alpha_shape
        alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy()
        if post_process:
            alpha_pred = utils.postprocess(alpha_pred)
        alpha_pred = alpha_pred * 255
        alpha_pred = alpha_pred.astype(np.uint8)
        alpha_pred = alpha_pred[32:h + 32, 32:w + 32]

        return alpha_pred


def generator_tensor_dict(video_frame, mask_path, args, id):
    # read images
    image = video_frame
    mask = cv2.imread(mask_path, 0)
    if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        # cv2.imwrite(os.path.join(save_dir_resize_mask, str(id).zfill(5) + '.jpg'), mask)
        mask = (mask > 0).astype(np.float32)
    else:
        mask = (mask >= args.guidance_thres).astype(np.float32)  ### only keep FG part of trimap
    cv2.imwrite(os.path.join(save_dir_resize_mask, str(id).zfill(5) + '.jpg'), mask*200)
    # mask = mask.astype(np.float32) / 255.0 ### soft trimap
    cv2.imwrite(os.path.join(save_dir_v_frame,str(id).zfill(5)+'.jpg'),image)
    # cv2.imwrite(os.path.join(save_dir_resize_mask,str(id).zfill(5)+'.jpg'),mask)
    print("mask_grey_size:", mask.shape)
    print("video_frame_size:",video_frame.shape)
    sample = {'image': image, 'mask': mask, 'alpha_shape': mask.shape}

    # reshape
    h, w = sample["alpha_shape"]

    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32, 32), (32, 32), (0, 0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32, 32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32, pad_h + 32), (32, pad_w + 32), (0, 0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32, pad_h + 32), (32, pad_w + 32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # convert GBR images to RGB
    image, mask = sample['image'][:, :, ::-1], sample['mask']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    mask = np.expand_dims(mask.astype(np.float32), axis=0)

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(mask)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # add first channel
    sample['image'], sample['mask'] = sample['image'][None, ...], sample['mask'][None, ...]

    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)
    print('mask_video to matting:')

    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/lab505/srh/MGMatting/code-base/config/MGMatting-DIM-Fortrain.toml')
    parser.add_argument('--checkpoint', type=str, default='pretrain/latest_model_unzip.pth')
    parser.add_argument('--data_dir', type=str, default='/media/lab505/Toshiba/MiVOS-main/Selected_data/0005')
    parser.add_argument('--save_mode', type=str, default='video')
    parser.add_argument('--guidance-thres', type=int, default=128, help="guidance input threshold")
    parser.add_argument('--post-process', action='store_true', default=False, help='post process to keep the largest connected component')

    # Parse configuration
    print(parser)
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    # args.output = os.path.join(args.output, CONFIG.version + '_' + args.checkpoint.split('/')[-1])
    # utils.make_dir(args.output)

    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()


    # read video and frames,shape
    data_path=args.data_dir
    v_path=os.path.join(data_path, "video.mp4")
    v_cap=cv2.VideoCapture(v_path)
    v_fps=v_cap.get(cv2.CAP_PROP_FPS)
    v_frames=int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_width=int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height=int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask_path=os.path.join(data_path,'mask','00001.png')
    mask_origin=cv2.imread(mask_path)
    mask_origin_width=mask_origin.shape[1]
    mask_origin_height=mask_origin.shape[0]
    print(v_width)
    print(v_height)

    # create frame_dir for video frame, resized mask,pre_alpha
    save_dir_v_frame = os.path.join(data_path, "video_frame")
    if not os.path.exists(save_dir_v_frame):
        os.mkdir(save_dir_v_frame)
    save_dir_resize_mask = os.path.join(data_path, "mask_resize")
    if not os.path.exists(save_dir_resize_mask):
        os.mkdir(save_dir_resize_mask)
    save_dir_pred_alpha = os.path.join(data_path, "pred_alpha")
    if not os.path.exists(save_dir_pred_alpha):
        os.mkdir(save_dir_pred_alpha)

    # save path
    save_path_a = os.path.join(data_path,"alpha_pred.mp4")
    save_path_m = os.path.join(data_path,"mask.mp4")

    if args.save_mode == 'video':
        # init the writting handler of generated video and alpha ground truth
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        out_alpha_pred= cv2.VideoWriter(save_path_a, fourcc, v_fps, ( v_width, v_height))
        out_mask = cv2.VideoWriter(save_path_m, fourcc, v_fps, (mask_origin_width, mask_origin_height))


    # abstract video frames and process it
    for i in range(v_frames):
        print("frame number:", str(i).zfill(5))
        # read video frame
        f_ret,f=v_cap.read()
        if f_ret==False or f is None :
            break
        #  read mask frame based on frame_id
        m_path=os.path.join(data_path,'mask',str(i).zfill(5)+'.png')
        m = cv2.imread(m_path)

        print("mask:",m.shape,"width:",m.shape[1],"height:",m.shape[0])
        image_dict=generator_tensor_dict(f,m_path,args,i)
        alpha_pred=single_inference(model,image_dict,post_process=args.post_process)
        alpha_pred=cv2.cvtColor(alpha_pred,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(save_dir_pred_alpha, str(i).zfill(5)+'.jpg'), alpha_pred)
        print("alpha size:",alpha_pred.shape)

        out_mask.write(m)
        out_alpha_pred.write(alpha_pred)

    v_cap.release()
    out_mask.release()
    out_alpha_pred.release()
    print("successs")
















