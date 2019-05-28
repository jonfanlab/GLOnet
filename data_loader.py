import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging
import scipy.io as io



def random_x_translation(img):
	translation = int(torch.randint(0, img.size(-1), (1,)))
	out = torch.cat([img[..., translation:], img[..., :translation]], dim=-1)
	return out



class MetaDataset(Dataset):
	def __init__(self, data_dir):
		training_dataset = io.loadmat(data_dir, struct_as_record = False, squeeze_me = True)

		self.Data = training_dataset['TOdata']
		self.wc = training_dataset['wc']
		self.wspan = training_dataset['wspan']
		self.ac = training_dataset['ac']
		self.aspan = training_dataset['aspan']


	def __len__(self):
		return len(self.Data)


	def __getitem__(self, idx):
		#if torch.cuda.is_available():
		#   label = torch.cuda.FloatTensor([self.wavelengths[idx], self.angles[idx]])
		#else:
		#   label = torch.FloatTensor([self.wavelengths[idx], self.angles[idx]])

		label = torch.FloatTensor([(self.Data[idx].wavelength - self.wc)/self.wspan, (self.Data[idx].angle - self.ac)/self.aspan])

		img = random_x_translation(torch.from_numpy(self.Data[idx].pattern).unsqueeze(0)).type(torch.FloatTensor)
		img = (img-0.5)/0.5
		
		return img, label


def fetch_dataloader(data_dir, params):
	
	dataloader = DataLoader(MetaDataset(data_dir),
							batch_size=params.batch_size,
							shuffle=True)

	# record normalization parameters
	params.wc = dataloader.dataset.wc
	params.ac = dataloader.dataset.ac
	params.wspan = dataloader.dataset.wspan
	params.aspan = dataloader.dataset.aspan

	return dataloader


