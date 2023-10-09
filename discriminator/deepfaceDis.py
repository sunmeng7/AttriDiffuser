import os

from deepface import DeepFace
# from discriminator import DeepFace
import torch


def deepfaceCos(solo_file):
    # solo_file = '/home/sunmeng/ab/ab5_dis/discriminator/train'
    solo_list = os.listdir(solo_file)

    solo_list.sort(key=lambda x:int(x[:-4]))
    embeddings = []
    for image in solo_list:
        img_path = str(solo_file) + '/' + image
        emb = DeepFace.represent(img_path=img_path, model_name='VGG-Face', enforce_detection=False)
        embeddings.append(emb[0]['embedding'])

    embeddings = torch.Tensor(embeddings).cuda()    # 6,2622

    similarity = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1).cuda()
    # print(similarity)
    return similarity

# deepfaceCos('/home/sunmeng/ab/ab5_dis/discriminator/train')