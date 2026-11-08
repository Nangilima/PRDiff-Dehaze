import torch
from torchvision.transforms.functional import crop

from utils.image_range import to_bipolar_range


def compute_alpha_bar(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


def reverse_sampling(x, x_cond, seq, noise_net, cgnet, betas, eta=0.0):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            alpha_bar = compute_alpha_bar(betas, t.long())
            alpha_bar_next = compute_alpha_bar(betas, next_t.long())
            xt = xs[-1].to(x.device)

            predicted_noise = noise_net(xt, t, x_cond)
            x0_t = (xt - predicted_noise * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            x0_preds.append(x0_t.to("cpu"))

            c1 = eta * ((1 - alpha_bar / alpha_bar_next) * (1 - alpha_bar_next)
                        / (1 - alpha_bar)).sqrt()
            c2 = ((1 - alpha_bar_next) - c1 ** 2).sqrt()
            xt_next = alpha_bar_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * predicted_noise
            xs.append(xt_next.to("cpu"))

    return xs, x0_preds


def overlapping_reverse_sampling(x, x_cond, seq, noise_net, cgnet, betas, eta=0.0,
                                 corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        overlap_count = torch.zeros_like(x, device=x.device)
        for hi, wi in corners:
            overlap_count[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        guidance_accum = torch.zeros_like(x, device=x.device)

        if manual_batching:
            batch_size = 64
            cond_patches = torch.cat(
                [to_bipolar_range(crop(x_cond, hi, wi, p_size, p_size)) for hi, wi in corners],
                dim=0
            )
            for i in range(0, len(corners), batch_size):
                guidance, _ = cgnet(cond_patches[i:i + batch_size])
                for idx, (hi, wi) in enumerate(corners[i:i + batch_size]):
                    guidance_accum[0, :, hi:hi + p_size, wi:wi + p_size] += guidance[idx]

        guidance_avg = torch.div(guidance_accum, overlap_count)
        x0_preds = [guidance_avg.to("cpu")]
        xs = [x, guidance_avg.to("cpu")]

    return xs, x0_preds
