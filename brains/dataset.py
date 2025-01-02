import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset

propmt_dict = {'n02106662': 'german shepherd dog',
'n02124075': 'cat ',
'n02281787': 'lycaenid butterfly',
'n02389026': 'sorrel horse',
'n02492035': 'Cebus capucinus',
'n02504458': 'African elephant',
'n02510455': 'panda',
'n02607072': 'anemone fish',
'n02690373': 'airliner',
'n02906734': 'broom',
'n02951358': 'canoe or kayak',
'n02992529': 'cellular telephone',
'n03063599': 'coffee mug',
'n03100240': 'old convertible',
'n03180011': 'desktop computer',
'n03197337': 'digital watch',
'n03272010': 'electric guitar',
'n03272562': 'electric locomotive',
'n03297495': 'espresso maker',
'n03376595': 'folding chair',
'n03445777': 'golf ball',
'n03452741': 'grand piano',
'n03584829': 'smoothing iron',
'n03590841': 'Orange jack-o’-lantern',
'n03709823': 'mailbag',
'n03773504': 'missile',
'n03775071': 'mitten,glove',
'n03792782': 'mountain bike, all-terrain bike',
'n03792972': 'mountain tent',
'n03877472': 'pajama',
'n03888257': 'parachute',
'n03982430': 'pool table, billiard table, snooker table ',
'n04044716': 'radio telescope',
'n04069434': 'eflex camera',
'n04086273': 'revolver, six-shooter',
'n04120489': 'running shoe',
'n07753592': 'banana',
'n07873807': 'pizza',
'n11939491': 'daisy',
'n13054560': 'bolete'
}

lable_number_dict={
'[12]': 'n02106662',
'[39]': 'n02124075',
'[11]': 'n02281787',
'[0]': 'n02389026',
'[21]': 'n02492035',
'[35]': 'n02504458',
'[8]': 'n02510455',
'[3]': 'n02607072',
'[36]': 'n02690373',
'[18]': 'n02906734',
'[10]': 'n02951358',
'[15]': 'n02992529',
'[5]': 'n03063599',
'[24]': 'n03100240',
'[17]': 'n03180011',
'[34]': 'n03197337',
'[28]': 'n03272010',
'[37]': 'n03272562',
'[4]': 'n03297495',
'[25]': 'n03376595',
'[16]': 'n03445777',
'[30]': 'n03452741',
'[2]': 'n03584829',
'[14]': 'n03590841',
'[23]': 'n03709823',
'[20]': 'n03773504',
'[27]': 'n03775071',
'[6]': 'n03792782',
'[31]': 'n03792972',
'[26]': 'n03877472',
'[1]': 'n03888257',
'[22]': 'n03982430',
'[38]': 'n04044716',
'[29]': 'n04069434',
'[7]': 'n04086273',
'[13]': 'n04120489',
'[32]': 'n07753592',
'[19]': 'n07873807',
'[9]': 'n11939491',
'[33]': 'n13054560'
}

img_file_type = '.JPEG'

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

class Dataset(Dataset):
    def __init__(self, img_names_file,seq_file,labels_file):
        self.image_names = torch.load(img_names_file)
        self.seqs = torch.load(seq_file)
        self.labels = torch.load(labels_file)

    def __len__(self):
            return len(self.seqs)

    def __getitem__(self, idx):
        input_vec = torch.tensor(self.seqs[idx]).to("cuda")  # (440,128)

        img_label = self.image_names[idx].split("_")[0]  # img_label = n03452741
        img_path = "eegdata/imageNet_images/" + img_label + "/" + self.image_names[idx] + img_file_type  # img_path = n03452741_16744.jpeg
        image = Image.open(img_path).convert('RGB')

        img_transform_test = transforms.Compose([
            normalize,
            transforms.Resize((512, 512)),
            channel_last
        ])
        gt_image = np.array(image) / 255.0
        gt_image = img_transform_test(gt_image)  # (512,512,3)

        prompt = propmt_dict[lable_number_dict[str(self.labels[idx])]]  # 'grand piano'

        return input_vec, gt_image,self.image_names[idx],prompt