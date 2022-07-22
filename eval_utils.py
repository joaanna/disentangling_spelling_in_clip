import sys
import torch
import numpy as np


def compute_imagenet_accuracy(image, text, class_id):
    logits = image @ text.T
    try:
        equal = torch.argmax(logits, 1) == class_id
    except:
        equal = np.argmax(logits, 1) == class_id
    return sum(equal) / len(logits)

def get_acc(logits_per_image):
    def _get_acc(logits_per_image):
        N = logits_per_image.shape[0]
        img2txt = (np.argmax(logits_per_image, 1) == np.arange(N)).sum() / N
        txt2img = (np.argmax(logits_per_image.T, 1) == np.arange(N)).sum() / N
        return round(100 * img2txt, 2), round(100 * txt2img, 2)

    if type(logits_per_image) == np.ndarray:
        return _get_acc(logits_per_image)
    else:
        return _get_acc(logits_per_image.detach().cpu().numpy())



def get_all_retireval_acc(r, set, files, out_dict, suffix='', U=1000000,
                          tasks=['im_it_img_t', 'im_it_img_i', 'im_it_txt_t',
                                       'im_t_txt_t']):
    if r == 'real':
        ID = 1
    else:
        ID = 0

    if 'im_t_txt_t' in tasks:
        val_org_logits = files['text_image'][files['real_or_fake']==ID][:U] @ files['text_features'][files['real_or_fake']==ID][:U].T
        acc1, acc2 = get_acc(val_org_logits)
        print(f'Initial accuracy on {set} {r} set [TEXT]: img2txt {acc1}')
        out_dict.update({f'acc_im_t_txt_t_img2txt_{r}_{suffix}': acc1, f'acc_im_t_txt_t_txt2img_{r}_{suffix}': acc2})

    if 'im_it_img_t' in tasks:
        val_org_logits = files['image_with_text'][files['real_or_fake']==ID][:U] @ files['text_image'][files['real_or_fake']==ID][:U].T
        acc1, acc2 = get_acc(val_org_logits)
        print(f'Initial accuracy on {set} {r} set [IMG+TXT / TXT IMAGE]: img2txt {acc1} ')
        out_dict.update({f'acc_im_it_img_t_img2txt_{r}_{suffix}': acc1, f'acc_im_it_img_t_txt2img_{r}_{suffix}': acc2})

    if 'im_it_img_i' in tasks:
        val_org_logits = files['image_with_text'][files['real_or_fake']==ID][:U] @ files['image'][files['real_or_fake']==ID][:U].T
        acc1, acc2 = get_acc(val_org_logits)
        print(f'Initial accuracy on {set} {r} set [IMG+TXT / IMG]: img2txt {acc1} ')
        out_dict.update({f'acc_im_it_img_i_img2txt_{r}_{suffix}': acc1, f'acc_im_it_img_i_txt2img_{r}_{suffix}': acc2})

    if 'im_it_txt_t' in tasks:
        val_org_logits = files['image_with_text'][files['real_or_fake']==ID][:U] @ files['text_features'][files['real_or_fake']==ID][:U].T
        acc1, acc2 = get_acc(val_org_logits)
        print(f'Initial accuracy on {set} {r} set [IMG+TXT / TEXT]: img2txt {acc1}')
        out_dict.update({f'acc_im_it_txt_t_img2txt_{r}_{suffix}': acc1, f'acc_im_it_txt_t_txt2img_{r}_{suffix}': acc2})
    return out_dict





def compute_original_accuracies(set, files, U, suffix='',
                                tasks=['im_it_img_t', 'im_it_img_i', 'im_it_txt_t',
                                       'im_t_txt_t', 'im_it_txt_i', 'im_i_txt_i']):
    out_dict = {}
    if 'im_i_txt_i' in tasks:
        acc1 = compute_imagenet_accuracy(files['image'], files['text_class_features'], files['class_id'])
        print(f'Initial accuracy on {set} set [IMAGENET]: {100*acc1}')
        out_dict.update({'acc_im_i_txt_i': acc1*100})

    if 'im_it_txt_i' in tasks:
        acc1 = compute_imagenet_accuracy(files['image_with_text'], files['text_class_features'], files['class_id'])
        print(f'Initial accuracy on {set} set [IMG+TXT / IMG CLASS]: {100*acc1}')
        out_dict.update({'acc_im_it_txt_i': acc1*100})

    out_dict = get_all_retireval_acc('real', set, files, out_dict, U=U, suffix=suffix, tasks=tasks)
    out_dict = get_all_retireval_acc('fake', set, files, out_dict, suffix=suffix, tasks=tasks)

    return out_dict

def get_subspace(clip_subspace_path):
    W = torch.load(clip_subspace_path)['W'].data.cpu().numpy()
    return W

def project(a, W):
    real_img_projected = a @ W
    real_img_projected /= np.linalg.norm(real_img_projected, axis=1)[:, None]
    return real_img_projected
