import os
import shutil

import torch
from torch.autograd import Variable
import tensorboardX


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


def add_scalars(dir, logfile, writer):

    with open(os.path.join(dir, logfile), 'r') as f:
        for line in f:
            parts = line.split('|')

            if len(parts) < 2 or 'end' not in parts[1]:
                continue

            step = int(parts[1].strip().split()[-1])
            loss_parts = parts[-2].strip().split()
            loss_val = float(loss_parts[-1].strip())
            ppl_parts = parts[-1].strip().split()
            ppl_val = float(ppl_parts[-1].strip())

            if 'mini' in parts[-1]:
                scalar_append = ' mini'
            else:
                scalar_append = ''
            writer.add_scalar('valid loss' + scalar_append, loss_val, step)
            writer.add_scalar('valid ppl' + scalar_append, ppl_val, step)


def add_embeddings(dir, model_file, labels_file, writer):

    model = torch.load(os.path.join(dir, model_file), map_location='cpu')
    labels = []
    with open(labels_file, 'r') as f:
        for label in f:
            labels.append(label.strip())
    writer.add_embedding(model.encoder.weight, metadata=labels)


def add_histograms(dir, model_file, writer, step=0):

    model = torch.load(os.path.join(dir, model_file), map_location='cpu')
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, step)


def plot_experiments(experiments_dir):

    labels_file = 'labels.txt'
    for dir in os.listdir(experiments_dir):
        writing_dir = 'logs/' + dir.split('/')[-1]
        writer = tensorboardX.SummaryWriter(writing_dir)

        experiment_dir = os.path.join(os.path.realpath(experiments_dir), dir)
        add_scalars(experiment_dir, 'log.txt', writer)
        add_embeddings(experiment_dir, 'model_mini.pt', labels_file, writer)
        add_histograms(experiment_dir, 'model_mini.pt', writer)

        writer.close()


def main():

    cwd = os.getcwd()
    plot_experiments(experiments_dir=os.path.join(cwd, 'experiments'))


if __name__ == '__main__':
    main()
