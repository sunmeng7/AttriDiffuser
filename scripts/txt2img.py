import argparse, os, sys, glob
from itertools import islice
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        # default="He has arched eyebrows, a beard and a receding hairline. He has a mustache and gray hair.",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="../data/output/ab5_celeba_644"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",  # 迭代次数
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=128,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=128,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",  # 多样性
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        # default='/home/sunmeng/413_ldm/latent-diffusion/data/val_txt_256.txt',
        default='/home/sunmeng/ab/ab2_asa_iaa/data/644',
        help="if specified, load prompts from this file, separated by newlines",
    )
    opt = parser.parse_args()

    config = OmegaConf.load(
        "/home/sunmeng/ab/ab5_dis/logs/2023-08-17_tys_ab5_0.1_celeba/configs/2023-08-21-project.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
        # "/home/sunmeng/ab/ab5_dis/logs/2023-08-16_tys_ab5_small_0.1/configs/2023-08-16-project.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config,
                                   "/home/sunmeng/ab/ab5_dis/logs/2023-08-17_tys_ab5_0.1_celeba/checkpoints/last.ckpt")  # TODO: check path
                                   # "/home/sunmeng/ab/ab5_dis/logs/2023-08-16_tys_ab5_small_0.1/checkpoints/last.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    if not opt.from_file:
        prompt = opt.prompt
        data = [opt.n_samples * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, 'r') as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(1)]
            data = list(chunk(data, opt.n_samples))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # label_list = ['black hair', 'wavy hair', 'bushy eyebrows', 'narrow eyes',
    #               'bags under eyes', 'high cheekbones', 'goatee', 'straight hair',
    #               'arched eyebrows', 'mustache', 'receding hairline', 'gray hair',
    #               'chubby', 'brown hair', 'pointy nose', 'beard', 'rosy cheeks',
    #               'heavy makeup', 'wearing lipstick', 'earrings', 'young',
    #               'big lips', 'pale skin', 'blond hair', 'hat', 'bangs',
    #               'necklace', 'double chin', 'big nose',
    #               'sideburns', 'necktie', 'oval face', 'eyeglasses', 'bald']

    label_list = ['black hair', 'wavy hair', 'bushy eyebrows', 'narrow eyes',
                  'bags under eyes', 'high cheekbones', 'goatee', 'straight hair',
                  'arched eyebrows', 'mustache', 'receding hairline', 'gray hair',
                  'chubby', 'brown hair', 'pointy nose', 'beard', 'rosy cheeks',
                  'heavy makeup', 'wearing lipstick', 'earrings', 'young',
                  'big lips', 'pale skin', 'blond hair', 'hat', 'bangs', 'attractive',
                  'necklace', 'double chin', 'big nose', 'mouth slightly open', '5\'o clock shadow',
                  'sideburns', 'necktie', 'oval face', 'eyeglasses', 'bald', 'smiling']

    all_samples = list()
    # linears = torch.nn.Sequential(
    #     torch.nn.Linear(512, 512),
    #     torch.nn.SiLU(),
    #     torch.nn.Linear(512, 768),
    # ).cuda()
    with torch.no_grad():
        with model.ema_scope():
            prompts = data[0]
            opts = []
            for i in range(len(prompts)):
                optss = []
                for j in label_list:
                    if j in prompts[0]:
                        atr = model.get_learned_conditioning(j).narrow(1, 0, 1).squeeze(1)
                        optss.append(atr)
                count = 0
                while len(optss) < 14:
                # while len(optss) < 34:
                    optss.append(optss[count])
                    count += 1
                optss = torch.cat(optss, dim=0).unsqueeze(0)
                opts.append(optss)
            opts = torch.cat(opts)
            # ccc = opts.float()
            # opts = linears(ccc)
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(opt.n_samples * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    # 3 -- channel  shape = [3, opt.H//8, opt.W//8]   f=8 -- sample 下采样因子
                    shape = [3, opt.H // 4, opt.W // 4]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps, opts=opts,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        # Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        prompts[0] = prompts[0][:50]
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f'{base_count:04}-{prompts[0].replace(" ", "-")}.png'))
                        base_count += 1
                    all_samples.append(x_samples_ddim)

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
