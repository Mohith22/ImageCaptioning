#Computer Vision - Course Project
#Instructor - Prof. Rob Fergus
#Image Captioning using ResNet as Encoder and LSTM as Decoder
#Dataset: MSCOCO (Microsoft Common Object in Context)
#Author: Mohith Damarapati | md4289 | N10540205 | New York University

#To retrieve back the vocabulary set
import pickle
#Data Loader
from load_data import load_data
#Torch and Torchvision
import torch
from torchvision import transforms
#Plot Losses
import matplotlib.pyplot as plt 
#Model File
import model 
#NLP Tool Kit
import nltk
#To compute BLEU Score
from nltk.translate.bleu_score import sentence_bleu
#Pycoco tools API
from pycocotools.coco import COCO
#Image Library
from PIL import Image
#To run on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn

lstmLayers = 3
lstmHiddenStates = 512
wordEmbeddings = 256
epochs = 5
batchSize = 64
learningRate = 0.001

def cap_reverse(caption, length):
	caption=caption.numpy()
	caps = []
	for i in range(length):
		caps.append(vocabularySet2[caption[i]])
		if(vocabularySet2[caption[i]] == '<e>'):
			return caps
	return caps


with open('vocabSet.pkl', 'rb') as f:
			vocabularySet = pickle.load(f)

print("Loaded Vocabulary Set")

with open('vocabSet2.pkl', 'rb') as f:
			vocabularySet2 = pickle.load(f)

print("Loaded Reverse Vocabulary Set")

modelsPath = "LSTM3Models/modelslstm3/"
imagesPath = "../data/val2014/"
captionsPath = "../data/annotations/captions_val.json"

cnnEn = model.EncoderCNN(wordEmbeddings).eval()  
lstmDe = model.DecoderRNN(wordEmbeddings, lstmHiddenStates, len(vocabularySet), lstmLayers)
cnnEn = cnnEn.to(device)
lstmDe = lstmDe.to(device)

valData = COCO(captionsPath)

cnnEn.load_state_dict(torch.load(modelsPath + "encoder_4" + ".ckpt"))
lstmDe.load_state_dict(torch.load(modelsPath+ "decoder_4"+".ckpt"))

#Exploiting Pycocotools to get insights about data
print("Total Annotations: " + str(len(valData.anns.keys())))
print("Total Images: " + str(len(valData.imgs.keys())))

#Visualise 
print(valData.imgToAnns[393212])

for (i, key) in enumerate(valData.imgToAnns.keys()):
	origCaptionSet = []
	for rec in valData.imgToAnns[key]:
		origCaptionSet.append(rec['caption'])

	break

#Print Lenght of Val Dataset
print(len(valData.imgs))
print(len(vocabularySet))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), 
							 (0.229, 0.224, 0.225))])






for (i, key) in enumerate(valData.imgs.keys()):
	path = (valData.imgs[key]['file_name'])
	print(path)
	image = Image.open(imagesPath + str(path)).convert('RGB')
	image = image.resize([224, 224], Image.LANCZOS)
	
	if transform is not None:
		image = transform(image).unsqueeze(0)
	
	origCaptionSet = []
	for rec in valData.imgToAnns[key]:
		#Convert string to list of words
		#print(rec)
		captions_wordlist = nltk.tokenize.word_tokenize(str(rec['caption']).lower())
		captions_wordlist = ['<s>'] + captions_wordlist 
		captions_wordlist.append('<e>')
		#print(captions_wordlist)
		origCaptionSet.append(captions_wordlist)

	image = image.to(device)

	cnnFeatures = cnnEn(image)
	genCaption = lstmDe.sample(cnnFeatures)
	genCaption = genCaption[0].cpu()

	genCaption = cap_reverse(genCaption, len(genCaption))

	print(genCaption)
	print(origCaptionSet) 

	bleu1Score = sentence_bleu(origCaptionSet, genCaption, weights=(1,0,0,0))
	bleu2Score = sentence_bleu(origCaptionSet, genCaption, weights=(0.5,0.5,0,0))
	bleu3Score = sentence_bleu(origCaptionSet , genCaption, weights=(0.33,0.33,0.33,0)) 
	bleu4Score = sentence_bleu(origCaptionSet , genCaption) 

	


	print("Bleu4:")
	print(bleu4Score)

	print("Bleu3")
	print(bleu3Score)

	print("Bleu2")
	print(bleu2Score)

	print("Bleu1")
	print(bleu1Score)
				


	

