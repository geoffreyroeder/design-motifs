## Code accompanying the paper Design Motifs for Generative Design
## Provided for research use only

from __future__ import division, absolute_import
import sys
import matplotlib
import argparse
import lasagne
from helpers.data import clip, write_model
from tqdm import tqdm
from helpers.mnist import load_mnist
from lasagne.updates import adam
import lasagne.layers as layers
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano
import theano.tensor as T
theano.config.compute_test_value = 'warn'  # 'warn' runs test values
matplotlib.use('Agg')


tanh = lasagne.nonlinearities.tanh
sigmoid = lasagne.nonlinearities.sigmoid
linear = lasagne.nonlinearities.linear
relu = lasagne.nonlinearities.rectify
softmax = lasagne.nonlinearities.softmax
NETWORK_DIM = 1024
LABEL_DIM = 1
# LABEL_DIM = 10  # e.g. labelled MNIST
DATA_DIM = 784
SAVEPATH = './'
np.random.seed(42)
srng = RandomStreams(42)
shared = lambda X: theano.shared(np.asarray(X, dtype=theano.config.floatX))


def unpack_params(params, latent_dim):
    return params[:, :latent_dim], params[:, latent_dim:]


def encoder_l(latent_dim, input_var=None):
    # input is concatenation of MNIST digit and one-hot encoded label
    input = layers.InputLayer(shape=(None, DATA_DIM+LABEL_DIM),
                              input_var=input_var)
    h1 = lasagne.layers.DenseLayer(input, NETWORK_DIM, nonlinearity=tanh)
    h2 = lasagne.layers.DenseLayer(h1, NETWORK_DIM, nonlinearity=tanh)
    h3 = lasagne.layers.DenseLayer(h2, NETWORK_DIM, nonlinearity=tanh)
    mu = lasagne.layers.DenseLayer(h3, latent_dim, nonlinearity=linear)
    log_std = lasagne.layers.DenseLayer(h3, latent_dim, nonlinearity=linear)
    return mu, log_std


def encoder_u(latent_dim, input_var=None):
    # input is concatenation of MNIST digit and one-hot encoded label
    input = layers.InputLayer(shape=(None, DATA_DIM), input_var=input_var)
    h1 = lasagne.layers.DenseLayer(input, NETWORK_DIM, nonlinearity=tanh)
    h2 = lasagne.layers.DenseLayer(h1, NETWORK_DIM, nonlinearity=tanh)
    h3 = lasagne.layers.DenseLayer(h2, NETWORK_DIM, nonlinearity=tanh)
    mu = lasagne.layers.DenseLayer(h3, latent_dim, nonlinearity=linear)
    log_std = lasagne.layers.DenseLayer(h3, latent_dim, nonlinearity=linear)
    return mu, log_std


def generator_x(n_hidden, input_var=None):
    # parameterize p(x | z) network
    input = layers.InputLayer(shape=(None, n_hidden), input_var=input_var)
    h1 = lasagne.layers.DenseLayer(input, NETWORK_DIM, nonlinearity=tanh)
    h2 = lasagne.layers.DenseLayer(h1, NETWORK_DIM, nonlinearity=tanh)
    h3 = lasagne.layers.DenseLayer(h2, NETWORK_DIM, nonlinearity=tanh)
    return lasagne.layers.DenseLayer(h3, DATA_DIM, nonlinearity=sigmoid)


def linear_regression(input_dim, input_var=None):
    input = lasagne.layers.InputLayer(shape=(None, input_dim),
                                      input_var=input_var)
    return lasagne.layers.DenseLayer(input, 1, nonlinearity=linear)


def multiclass_logistic(input_dim, num_classes, input_var=None):  # i.e., gen_t
    # add L2 regularization
    input = lasagne.layers.InputLayer(shape=(None, input_dim),
                                      input_var=input_var)
    return lasagne.layers.DenseLayer(input, num_classes, nonlinearity=softmax)


def sample_q(mu, log_std, s=42):
        if "gpu" in theano.config.device:
            rng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=s)
        else:
            rng = T.shared_randomstreams.RandomStreams(seed=s)
        eps = rng.normal(mu.shape)
        z = mu + T.exp(0.5 * log_std) * eps
        return z


def lower_bound_l(enc, gen_x, gen_t, data, targets, scale_l):
    X_and_t = T.concatenate([data, targets], axis=1)
    mu, log_std = layers.get_output(enc, X_and_t)
    zIxt = sample_q(mu, log_std)
    targetsIz = layers.get_output(gen_t, zIxt)  # ? x 1
    xIz = layers.get_output(gen_x, zIxt)
    log_ptIz = gaussian_log_likelihood(data, 0.03, targetsIz, 1)
    log_pxIz = bernoulli_log_likelihood(xIz, data)
    KL_qIIp = -0.5 * T.sum(1 + log_std - mu ** 2 - T.exp(log_std), axis=1)
    loss = - T.mean(log_pxIz + log_ptIz - KL_qIIp)  # N_l * mean(lower_bound_l)
    recons_loss = T.mean(T.sum(T.square(data - xIz), axis=1))
    return loss, recons_loss


def lower_bound_u(enc, gen, data, scale_u):
    mu, log_std = layers.get_output(enc, data)
    latents = sample_q(mu, log_std)
    generated = layers.get_output(gen, latents)
    log_pxIz = bernoulli_log_likelihood(generated, data)
    KL_qIIp = -0.5 * T.sum(1 + log_std - mu ** 2 - T.exp(log_std), axis=1)
    loss = -T.mean(log_pxIz - KL_qIIp)
    recons_loss = T.mean(T.sum(T.square(data - generated), axis=1))
    return loss, recons_loss


def main(step_size, batch_size, n_epochs, save, plot, latent_dim,
         proportion_labelled):
    # data, targets = load_mnist('train', label=True) # class labels
    data, _ = load_mnist('train', label=True)
    # brightness target function
    targets = data.reshape(-1, 784).sum(axis=-1, keepdims=True)

    n = data.shape[0]
    n_valid = 10000
    n_train = n - n_valid
    n_batches = n_train//batch_size
    bs_l = int(batch_size*proportion_labelled)
    bs_u = int(batch_size*(1-proportion_labelled))

    # calculate weightings for sup and unsup losses
    scale_l = bs_l / batch_size
    scale_u = bs_u / batch_size
    assert(scale_l + scale_u == 1)

    X_train = data[:n_train, :].reshape(-1, 784)
    X_valid = data[n_train:, :].reshape(-1, 784)
    t_train = targets[:n_train]
    t_valid = targets[n_train:]

    n_u = int(n_train * (1 - proportion_labelled))

    scope = 'main/'  # track scope for debugging
    X_u = T.fmatrix(scope + 'X_unlabelled')
    X_l = T.fmatrix(scope + 'X_labelled')
    t_l = T.fmatrix(scope + 'targets')
    X_u.tag.test_value = np.random.rand(100, DATA_DIM).astype(np.float32)
    X_l.tag.test_value = np.random.rand(100, DATA_DIM).astype(np.float32)
    t_l.tag.test_value = np.random.randint(0, 783, (100, 1)).astype(np.float32)

    # construct unsupervised lower bound
    # q(z | x) network
    enc_zIx = encoder_u(latent_dim) 
   
    # p(x | z) network
    gen_xIz = generator_x(latent_dim) 
    loss_u, recons_loss_u = lower_bound_u(enc_zIx, gen_xIz, X_u, scale_u)
    
    # construct supervised lower bound
    # p(t | z) network
    enc_zIxt = encoder_l(latent_dim)
    # gen_tIz = multiclass_logistic(latent_dim, num_classes)
    gen_tIz = linear_regression(latent_dim)  # input is ? x latent_dim

    if proportion_labelled > 0:
        loss_l, recons_loss_l = lower_bound_l(enc_zIxt, gen_xIz, gen_tIz,
                                              X_l, t_l, scale_l)
    else:
        loss_l = 0
        recons_loss_l = 0

    # SSVAE lower bound
    loss = (loss_u + loss_l)  #/ batch_size
    valid_loss = (loss_u + loss_l)  #/ n_valid
    recons_loss = (recons_loss_l + recons_loss_u) / batch_size
    
    valid_loss = [loss, recons_loss]

    # build symbolic sampler for learned generator
    noise = T.fmatrix()
    noise.tag.test_value = np.random.rand(100, latent_dim).astype(np.float32)
    gen_im = layers.get_output(gen_xIz, noise)
    gen_fun = theano.function([noise], gen_im)

    # learn SSVAE parameters
    idx = T.iscalar()
    idx.tag.test_value = 0
    params = layers.get_all_params(enc_zIx) +\
             layers.get_all_params(gen_xIz)

    if proportion_labelled > 0:
        params += layers.get_all_params(enc_zIxt) +\
                 layers.get_all_params(gen_tIz)

    # split train data into sup/unsup
    optimize_params = adam(loss, params, learning_rate=step_size)
    train_u = shared(X_train[:n_u, :])
    train_l = shared(X_train[n_u:, :])
    train_t = shared(t_train[n_u:])

    train_dict = {X_u: train_u[idx*bs_u:(idx+1)*bs_u],
                  X_l: train_l[idx*bs_l:(idx+1)*bs_l],
                  t_l: train_t[idx*bs_l:(idx+1)*bs_l]}

    # split validation data into sup/unsup
    n_v_u = int(n_valid * (1 - proportion_labelled))
    valid_u = shared(X_valid[:n_v_u, :])
    valid_l = shared(X_valid[n_v_u:, :])
    valid_t = shared(t_valid[n_v_u:])
        
    valid_dict = {X_u: valid_u, X_l: valid_l,
                  t_l: valid_t}
   
    # build function to forward propagate data and update weights in SSVAE
    train = theano.function([idx], [loss, recons_loss], updates=optimize_params,
                            givens=train_dict, on_unused_input='warn')
    validate = theano.function([], valid_loss, givens=valid_dict,
                               on_unused_input='warn')

    losses = {"train": [], "valid": []}
    train_loss = None
    model_name='t:reg_bMNISTz'+str(latent_dim)+'bs'+\
               str(batch_size)+'lr'+str(step_size)+\
               'l%'+str(proportion_labelled) + 'joint'
    for e in tqdm(range(n_epochs)):
        for i in tqdm(range(n_batches)):
            train_loss = train(i)
        # sanity test
        if np.isnan(train_loss[0]):
            print ("NaN detected!")
            sys.stdout.flush()
            break
        valid_loss = validate()
        losses["train"].append(train_loss[0])
        losses["valid"].append(valid_loss[0])

        # write progress to STDOUT
        if e%5 == 0:
            print("epoch "+str(e)+" train_loss: " + str(train_loss[0])
                  + "  train_recon_loss: "+str(train_loss[1]))
            sys.stdout.flush()
            print("   valid_loss: "+str(valid_loss[0]) + " valid_recon_loss: "
                  + str(valid_loss[1]))
            sys.stdout.flush()

        # checkpoint model parameters
        if save and (e+1)%5 == 0:
            write_model(gen_xIz, model_name + 'genxIz', e, SAVEPATH)
            write_model(enc_zIx, model_name + 'enczIx', e, SAVEPATH)
            write_model(enc_zIxt, model_name + 'enczIxt', e, SAVEPATH)
            write_model(gen_tIz, model_name + 'gentIz', e, SAVEPATH)
        

def gaussian_log_likelihood(X_data, fixed_var, generated, input_dim):
    return -T.sum(T.square(generated - X_data), [-1]) / (2 * fixed_var) \
               - input_dim / 2 * np.log(2 * np.pi) \
               - input_dim / 2 * np.log(fixed_var)


def bernoulli_log_likelihood(generated, input):
    return -T.sum(T.nnet.binary_crossentropy(clip(generated), input),
                  axis=1)


def get_args(parser):
    parser.add_argument("--step_size", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--latent_dim", default=2, type=int)
    parser.add_argument("--save", default=1, type=int)
    parser.add_argument("--plot", default=1, type=int)
    parser.add_argument("--genx", default=None, type=str)
    parser.add_argument("--gent", default=None, type=str)
    parser.add_argument("--proportion_labelled", default=0.1, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args(argparse.ArgumentParser())
    print("no pretrained files found")
    sys.exit(-1)
    main(args.step_size, args.batch_size, args.n_epochs, args.save, args.plot,
         args.latent_dim, args.proportion_labelled)
