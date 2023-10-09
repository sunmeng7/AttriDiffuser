import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            # nn.Linear( self.in_dim + self.position_dim, 512),
            # nn.Linear( 512, 512),
            # nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, positive_embeddings):
        list = []
        # count = 0
        for item in positive_embeddings:
            if item[0].shape[0] < 14:
                mul = 14 // item[0].shape[0]
                yu = 14 % item[0].shape[0]
                # data = item[0]
                if mul > 1:
                    item[0] = torch.repeat_interleave(item[0], mul, dim=0)
                item[0] = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                list.append(item[0].narrow(1, 0, 1).squeeze())
            elif item[0].shape[0] == 14:
                list.append(item[0].narrow(1, 0, 1).squeeze())
                # data = torch.cat((data, item[0][0:yu, :, :]), dim=0)
                ## print(mul)
                # if mul == 2:
                #     yu = 14 % item[0].shape[0]
                #     item[0] = torch.cat((item[0], item[0]), dim=0)
                #     data = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                # elif mul == 1:
                #     data = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                # elif mul == 3:
                #     item[0] = torch.cat((item[0], item[0], item[0]), dim=0)
                #     data = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                # elif mul == 4:
                #     item[0] = torch.cat((item[0], item[0], item[0], item[0]), dim=0)
                #     data = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                # elif mul == 5:
                #     item[0] = torch.cat((item[0], item[0], item[0], item[0], item[0]), dim=0)
                #     data = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                # elif mul == 6:
                #     item[0] = torch.cat((item[0], item[0], item[0], item[0], item[0], item[0]), dim=0)
                #     data = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                # elif mul == 7:
                #     item[0] = torch.cat((item[0], item[0], item[0], item[0], item[0], item[0], item[0]), dim=0)
                #     data = torch.cat((item[0], item[0][0:yu, :, :]), dim=0)
                # list.append(data.narrow(1, 0, 1).squeeze())
                # item[0] = data
        data = torch.stack(list)

            #
        #     while len(item) < 14:  # 列表
        #         item.append(item[count])
        #         count += 1
        #     data = torch.cat(item, dim=0)
        #     list.append(data.narrow(1, 0, 1).squeeze())  # data.narrow(1,0,1).squeeze() # 6个14,1024
        #     count = 0
        # data = torch.stack(list)    # 6,14,1024

        # 2,34,512
        # B, N, S, _ = data.shape
        B, N, _ = data.shape
        objs = data
        # objs = self.linears(data)
        # assert objs.shape[0] == 6
        assert objs.shape == torch.Size([B, N, self.out_dim])
        # assert objs.shape == torch.Size([B,N,S,self.out_dim])
        return objs  # 2,30,768
