import os
import shutil
import torch

from torch.autograd import Variable


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len])
    # target = Variable(source[i+1:i+1+seq_len].view(-1))
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, path, append='', finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model' + append + '.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model' + append + '.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))


def negative_targets_torch(true_targets, ntokens, k, probs=None):

    if probs is None:
        sampling_probs = torch.ones(true_targets.size(0), ntokens).to(true_targets).to(torch.float)
    else:
        sampling_probs = probs.repeat(true_targets.size(0), 1)

    idx_x = list(range(true_targets.size(0)))
    sampling_probs[idx_x, true_targets] = 0

    negative_targets = torch.multinomial(sampling_probs, k).to(true_targets)
    result = torch.cat((true_targets.view(-1, 1), negative_targets), dim=1)

    if probs is not None:
        p_noise = probs[result]
    else:
        p_noise = None

    return result, p_noise
