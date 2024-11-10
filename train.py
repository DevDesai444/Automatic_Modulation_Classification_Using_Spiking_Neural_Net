import torch
import numpy as np
import os
import argparse
import datetime

from dcll.pytorch_libdcll import device
from dcll.experiment_tools import mksavedir, save_source, annotate
from dcll.pytorch_utils import grad_parameters, named_grad_parameters, NetworkDumper, tonumpy
from networks import ConvNetwork, ReferenceConvNetwork, load_network_spec

from data.utils import to_one_hot
from data.load_radio_ml import get_radio_ml_loader as get_loader
from data.utils import iq2spiketrain as to_spike_train

def parse_args():
    parser = argparse.ArgumentParser(description='DCLL')
    parser.add_argument('--radio_ml_data_dir', type=str, default='2018.01',
                        help='path to the folder containing the RadioML HDF5 file(s)')
    parser.add_argument('--min_snr', type=int, default=6,
                        metavar='N', help='minimum SNR (inclusive) to use during data loading')
    parser.add_argument('--max_snr', type=int, default=30,
                        metavar='N', help='maximum SNR (inclusive) to use during data loading')
    parser.add_argument('--per_h5_frac', type=float, default=0.5,
                        metavar='N', help='fraction of each HDF5 data file to use')
    parser.add_argument('--train_frac', type=float, default=0.9,
                        metavar='N', help='train split (1-TRAIN_FRAC is the test split)')
    parser.add_argument('--network_spec', type=str, default='networks/radio_ml_conv.yaml',
                        metavar='S', help='path to YAML file describing net architecture')
    parser.add_argument('--I_resolution', type=int, default=128,
                        metavar='N', help='size of I dimension (used when representing I/Q plane as image)')
    parser.add_argument('--Q_resolution', type=int, default=128,
                        metavar='N', help='size of Q dimension (used when representing I/Q plane as image)')
    parser.add_argument('--I_bounds', type=float, default=(-1, 1),
                        nargs=2, help='range of values to represent in I dimension of I/Q image')
    parser.add_argument('--Q_bounds', type=float, default=(-1, 1),
                        nargs=2, help='range of values to represent in Q dimension of I/Q image')
    parser.add_argument('--burnin', type=int, default=50,
                        metavar='N', help='burnin')
    parser.add_argument('--batch_size', type=int, default=64,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=64,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--n_steps', type=int, default=10000,
                        metavar='N', help='number of steps to train')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='S', help='random seed')
    parser.add_argument('--n_test_interval', type=int, default=20,
                        metavar='N', help='how many steps to run before testing')
    parser.add_argument('--n_test_samples', type=int, default=128,
                        metavar='N', help='how many test samples to use')
    parser.add_argument('--n_iters', type=int, default=1024, metavar='N',
                        help='for how many ms do we present a sample during classification')
    parser.add_argument('--n_iters_test', type=int, default=1024, metavar='N',
                        help='for how many ms do we present a sample during classification')
    parser.add_argument('--optim_type', type=str, default='Adam',
                        metavar='S', help='which optimizer to use')
    parser.add_argument('--loss_type', type=str, default='SmoothL1Loss',
                        metavar='S', help='which loss function to use')
    parser.add_argument('--learning_rates', type=float, default=[1e-6],
                        nargs='+', metavar='N', help='learning rates for each DCLL slice')
    parser.add_argument('--alpha', type=float, default=.92,
                        metavar='N', help='Time constant for neuron')
    parser.add_argument('--alphas', type=float, default=.85,
                        metavar='N', help='Time constant for synapse')
    parser.add_argument('--alpharp', type=float, default=.65,
                        metavar='N', help='Time constant for refractory')
    parser.add_argument('--arp', type=float, default=0,
                        metavar='N', help='Absolute refractory period in ticks')
    parser.add_argument('--random_tau', type=bool, default=True,
                        help='randomize time constants in convolutional layers')
    parser.add_argument('--beta', type=float, default=.95,
                        metavar='N', help='Beta2 parameters for Adamax')
    parser.add_argument('--lc_ampl', type=float, default=0.5,
                        metavar='N', help='magnitude of local classifier init')
    parser.add_argument('--netscale', type=float, default=1.,
                        metavar='N', help='scale network size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    get_loader_kwargs  = {}
    to_st_train_kwargs = {}
    to_st_test_kwargs  = {}

    #set no of iterations
    n_iters = args.n_iters
    n_iters_test = args.n_iters_test

    #image dimensions
    im_dims = (1, args.Q_resolution, args.I_resolution)
    
    #2 classes
    target_size = 2

    # Set "get loader" kwargs
    get_loader_kwargs['data_dir'] = args.radio_ml_data_dir
    get_loader_kwargs['min_snr'] = args.min_snr
    get_loader_kwargs['max_snr'] = args.max_snr
    get_loader_kwargs['per_h5_frac'] = args.per_h5_frac
    get_loader_kwargs['train_frac'] = args.train_frac

    # Set "to spike train" kwargs
    for to_st_kwargs in (to_st_train_kwargs, to_st_test_kwargs):
        to_st_kwargs['out_w'] = args.I_resolution
        to_st_kwargs['out_h'] = args.Q_resolution
        to_st_kwargs['min_I'] = args.I_bounds[0]
        to_st_kwargs['max_I'] = args.I_bounds[1]
        to_st_kwargs['min_Q'] = args.Q_bounds[0]
        to_st_kwargs['max_Q'] = args.Q_bounds[1]
    to_st_train_kwargs['max_duration'] = n_iters
    to_st_train_kwargs['gs_stdev'] = 0
    to_st_test_kwargs['max_duration'] = n_iters_test
    to_st_test_kwargs['gs_stdev'] = 0

    # number of test samples: n_test * batch_size_test
    n_test = np.ceil(float(args.n_test_samples) /
                     args.batch_size_test).astype(int)
    n_tests_total = np.ceil(float(args.n_steps) /
                            args.n_test_interval).astype(int)

    #set optimizer and loss function
    opt = getattr(torch.optim, args.optim_type)
    opt_param = {
        'betas': [0.0, args.beta],
        'weight_decay': 10.0,
    }
    loss = getattr(torch.nn, args.loss_type)

    #load DCLL CNN based on parameters specified in  networks/radio_ml_conv.yaml
    burnin = args.burnin
    convs = load_network_spec(args.network_spec)
    net = ConvNetwork(args, im_dims, args.batch_size, convs, target_size,
                        act=torch.nn.Sigmoid(), loss=loss, opt=opt, opt_param=opt_param,
                        learning_rates=args.learning_rates, burnin=burnin)

    #network init before training starts
    net = net.to(device)
    net.reset(True)
    acc_test = np.empty([n_tests_total, n_test, len(net.dcll_slices)])

    #extract readable numpy train data from hdf5 files in dataset
    train_data = get_loader(args.batch_size, train=True, **get_loader_kwargs)
    gen_train = iter(train_data)
    gen_test = iter(get_loader(args.batch_size_test, train=False, **get_loader_kwargs))

    all_test_data = [next(gen_test) for i in range(n_test)]
    all_test_data = [(samples, to_one_hot(labels, target_size))
                     for (samples, labels) in all_test_data]
    
    for step in range(args.n_steps):
        if ((step + 1) % 1000) == 0:
            for i in range(len(net.dcll_slices)):
                net.dcll_slices[i].optimizer.param_groups[-1]['lr'] /= 2
            net.dcll_slices[-1].optimizer2.param_groups[-1]['lr'] /= 2
            print('Adjusting learning rates')

        #extract data 1 batch_size at a time (default = 512)
        try:
            input, labels = next(gen_train)
        except StopIteration:
            gen_train = iter(train_data)
            input, labels = next(gen_train)

        labels = to_one_hot(labels, target_size)

        #no of images, out of the 1024 timesteps per signal (1 timestep I/Q value is 1 image), sent into the network for training
        n_iters_sampled = n_iters
        to_st_train_kwargs['max_duration'] = n_iters_sampled

        #convert image to data to spike train
        input_spikes, labels_spikes = to_spike_train(input, labels,
                                                        **to_st_train_kwargs)
        input_spikes = torch.Tensor(input_spikes).to(device)
        labels_spikes = torch.Tensor(labels_spikes).to(device)

        #n_iters images generated from each of the 512 signals is sent into the network for training
        net.reset()
        net.train()
        for sim_iteration in range(n_iters_sampled):
            net.learn(x=input_spikes[sim_iteration],
                        labels=labels_spikes[sim_iteration])
            
        #Training accuracy acheived for each dcll slice / each layer
        acc_train = net.accuracy(labels_spikes)
        step_str = str(step).zfill(5)
        print('[TRAIN] Step {} \t Accuracy {} -----> Timestamp = {}'.format(step_str, acc_train,datetime.datetime.now()))

        # Test after every n_test_interval (default=10)
        if (step % args.n_test_interval) == 0:
            test_idx = step // args.n_test_interval
            for i, test_data in enumerate(all_test_data):

                #convert test data to spike train
                test_input, test_labels = to_spike_train(*test_data,
                                                            **to_st_test_kwargs)
                try:
                    test_input = torch.Tensor(test_input).to(device)
                except RuntimeError as e:
                    print('Exception: ' + str(e) +
                            '. Try to decrease your batch_size_test with the --batch_size_test argument.')
                    raise
                test_labels = torch.Tensor(test_labels).to(device)

                #send in a vector of batch size x no of images per signal and run for all images per signal
                #prediction is the class predicted max no of times in the 1024 images for each signal
                net.reset()
                net.eval()

                for sim_iteration in range(n_iters_test):
                    net.test(x=test_input[sim_iteration])

                acc_test[test_idx, i, :] = net.accuracy(test_labels)

            #print the accuracy
            acc = np.mean(acc_test[test_idx], axis=0)
            step_str = str(step).zfill(5)
            print('[TEST]  Step {} \t Accuracy {}'.format(step_str, acc))
