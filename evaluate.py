import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

def merge_imgs(folder):
    data_folder = 'results/' + folder
    output_size = (256, 256)
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            figure_folder = os.path.join('results/figure', folder, subfolder)
            os.makedirs(figure_folder, exist_ok=True)
            num_images = len(os.listdir(subfolder_path + '/input'))
            for index in range(num_images):
                images = []
                for name in ['input', 'label', 'mcg', 'ps']:
                    path = os.path.join(subfolder_path, name, f'{index:05d}.png')
                    image = Image.open(path).convert('RGB')
                    image = TF.resize(TF.to_tensor(image), output_size)
                    images.append(image)
                merged_image = vutils.make_grid(images, nrow=4, padding=4, pad_value=1.0)
                output_path = os.path.join(figure_folder, f'{index:05d}.png')
                vutils.save_image(merged_image, output_path)

def compute_FID(folder):
    data_folder = 'results/' + folder
    get_value = lambda s: float(s.readlines()[-1].split(':')[1].lstrip(' ').rstrip('\n'))
    res = {}
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)

        if os.path.isdir(subfolder_path):
            label_path = os.path.join(subfolder_path, 'label')
            ps_path = os.path.join(subfolder_path, 'ps')
            mcg_path = os.path.join(subfolder_path, 'mcg')
            ps_res = os.popen('python -m pytorch_fid {:} {:}'.format(label_path, ps_path))
            mcg_res = os.popen('python -m pytorch_fid {:} {:}'.format(label_path, mcg_path))
            res[subfolder] = {'ps': get_value(ps_res), 'mcg': get_value(mcg_res)}
    for k, v in res.items():
        print('{:<30}| PS: {:4.3f} | MCG: {:4.3f}'.format(k, v['ps'], v['mcg']))

def compute_LPIPS(folder):
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')
    res = {}
    data_folder = 'results/' + folder
    normalize = lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            num_images = len(os.listdir(subfolder_path + '/input'))
            res[subfolder] = {'ps': 0, 'mcg': 0}
            for index in range(num_images):
                image_func = lambda name: TF.to_tensor(Image.open(
                    os.path.join(subfolder_path, name, f'{index:05d}.png')).convert('RGB'))
                label_image = normalize(image_func('label'))
                ps_image = normalize(image_func('ps'))
                mcg_image = normalize(image_func('mcg'))
                res[subfolder]['ps'] += loss_fn_alex(label_image, ps_image).item() / num_images
                res[subfolder]['mcg'] += loss_fn_alex(label_image, mcg_image).item() / num_images
    for k, v in res.items():
        print('{:<30}| PS: {:4.3f} | MCG: {:4.3f}'.format(k, v['ps'], v['mcg']))

if __name__ == '__main__':
    compute_FID('test')

