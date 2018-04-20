from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#import theano.sandbox.linalg as T_linalg
from scipy.special import erfinv

import cPickle

from collections import OrderedDict

if theano.config.floatX == "float16":
    print ("using epsilon=1e-6")
    epsilon = 1e-6
else:
    epsilon = 1e-8

class VariationalAutoencoder:
    """
    This class implements a Doubly Variational Autoencoder

    Parameters
    --------------
    data: processed data class from helper.py
    encoder_architecture: List for size of layers of encoder
    decoder_architecture: List for size of layers of decoder
    n_latent: Number of latent variables
    n_patterns: Number of times the scale parameters should be tiled over the
                    final weights of the decoder
    batch_size: Mini-batch size
    encode_nonlinearity_type: Nonlinearity of encoder
    decode_nonlinearity_type: Nonlinearity of decoder
    final_decode_nonlinearity: Final nonlinearity of decoder
    sparsity: Sparsity type, using a noncentered reparameterization.
                Options include: logit, analytic, laplacian, horseshoe, ard
                See Ingraham and Marks, 2016 (https://arxiv.org/abs/1602.03807)
    global_scale: Global scale prior for sparsity: analytic, laplacian, horseshoe, ard
    logit_p: Global scale prior for logit sparsity
    logit_sigma: Prior sigma for scale prior for logit sparsity
    pattern_sigma: Prior sigma for variational weights on the final layer
    warm_up: Annealing schedule for KL (default 0)
    convolve_encoder: Include 1D conv on the input sequences
    convolve_patterns: Include 1D conv on the final decoder weights
                        Also known as the dictionary
    conv_encoder_size: Convolution size for input
    conv_decoder_size: Convolution size for dictionary
    output_bias: Include an output bias
    final_pwm_scale: Include inverse temperature parameter
    working_dir: directory to save and load parameters
    kl_scale: Scale of KL of latent variables, default 1.0
                Scale < 1.0 approaches a normal autoencoder
                Scale > 1.0 turns into beta-autoencoder (Higgins et al, 2016)
    learning_rate: Adam learning rate,
    b1: Adam b1 hyperparameter
    b2: Adam b2 hyperparameter
    random_seed: Random init seed

    Returns
    ------------
    None (Purpose of the file is to make callables for training and inference)

    """
    def __init__(self,
        data,
        encoder_architecture=[1500,1500],
        decoder_architecture=[100,500],
        n_latent=2,
        n_patterns=4,
        batch_size=100,
        encode_nonlinearity_type="relu",
        decode_nonlinearity_type="relu",
        final_decode_nonlinearity="sigmoid",
        sparsity="logit",
        global_scale=1.0,
        logit_p=0.01,
        logit_sigma=4.0,
        pattern_sigma=1.0,
        warm_up=0.0,
        convolve_encoder=False,
        convolve_patterns=True,
        conv_decoder_size=10,
        conv_encoder_size=10,
        output_bias=True,
        final_pwm_scale=False,
        working_dir=".",
        learning_rate=0.001,
        kl_scale=1.0,
        b1=0.9,
        b2=0.999,
        random_seed=42):

        # Directory for saving parameters
        self.working_dir = working_dir

        # Network architecture
        self.n_latent = n_latent
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture
        self.n_patterns = n_patterns
        # For cases that go from latent -> output
        if self.decoder_architecture != []:
            self.decoder_architecture[-1] = self.decoder_architecture[-1] \
                * self.n_patterns
        else:
            self.n_latent = n_latent * self.n_patterns

        self.seq_len = data.seq_len
        self.alphabet_size = data.alphabet_size
        self.encode_nonlinearity_type = encode_nonlinearity_type
        self.decode_nonlinearity_type = decode_nonlinearity_type

        self.decoder_type = "svi"
        self.output_bias = output_bias
        self.final_pwm_scale = final_pwm_scale

        # Sparsity hyperparameters
        self.sparsity = sparsity
        self.global_scale = \
            theano.shared(global_scale).astype(theano.config.floatX)
        self.inv_global_scale = theano.shared(1.0 \
            / global_scale).astype(theano.config.floatX)

        self.logit_p = logit_p

        # Calculate the mu with the given sigma and logit p
        #    to calculate the approximate mu
        self.logit_mu = theano.shared(np.sqrt(2.0) * logit_sigma \
            * erfinv(2.0 * logit_p - 1.0)).astype(theano.config.floatX)
        self.logit_sigma = \
            theano.shared(logit_sigma).astype(theano.config.floatX)

        # Algorithmic options
        self.final_decode_nonlinearity = final_decode_nonlinearity
        self.convolve_encoder = convolve_encoder
        self.conv_encoder_size = conv_encoder_size
        self.conv_decoder_size = conv_decoder_size
        self.convolve_patterns = convolve_patterns
        self.kl_scale = theano.shared(kl_scale).astype(theano.config.floatX)
        self.warm_up = theano.shared(warm_up).astype(theano.config.floatX)


        self.prng = np.random.RandomState(random_seed)

        # Learning parameters
        self.batch_size = batch_size
        self.b1 = theano.shared(b1).astype(theano.config.floatX)
        self.b2 = theano.shared(b2).astype(theano.config.floatX)
        self.learning_rate = \
            theano.shared(learning_rate).astype(theano.config.floatX)
        self.t = theano.shared(0.0).astype(theano.config.floatX)

        # Parameter initialization
        self.sigma_init = 0.01
        self.logsig_init = -5
        # Glorot initialization
        create_weight = lambda dim_input, dim_output: self.prng.normal(0.0, \
            np.sqrt(2.0 / (dim_input + dim_output)), \
            (dim_input, dim_output)).astype(theano.config.floatX)
        create_weight_zeros = lambda dim_input, dim_output: \
            np.zeros((dim_input, dim_output)).astype(theano.config.floatX)
        create_bias = lambda dim_output:  0.1 \
            * np.ones(dim_output).astype(theano.config.floatX)

        # Variational uncertainty
        create_weight_logsig = lambda dim_input, dim_output: self.logsig_init \
            * np.ones((dim_input, dim_output)).astype(theano.config.floatX)
        create_bias_logsig = lambda dim_output: self.logsig_init \
            * np.ones(dim_output).astype(theano.config.floatX)

        self.params = OrderedDict()
        self.variational_param_name_list = []
        self.variational_param_name_to_sigma = {}

        # Initialize encoder variables
        for layer_num, hidden_units in enumerate(self.encoder_architecture):
            w_name = "W_encode_"+str(layer_num)
            b_name = "b_encode_"+str(layer_num)

            if layer_num == 0:
                if self.convolve_encoder:

                    self.params["W_conv_encode"] = theano.shared( \
                        create_weight(self.alphabet_size, \
                        self.conv_encoder_size), \
                        name="W_conv_encode")

                    self.params[w_name] = theano.shared( \
                        create_weight(self.seq_len * self.conv_encoder_size, \
                        hidden_units), \
                        name=w_name)

                    self.params[b_name] = theano.shared( \
                        create_bias(hidden_units), \
                        name=b_name)

                else:
                    self.params[w_name] = theano.shared( \
                        create_weight(self.seq_len * self.alphabet_size, \
                        hidden_units), \
                        name=w_name)

                    self.params[b_name] = theano.shared( \
                        create_bias(hidden_units), \
                        name=b_name)

            else:
                prev_hidden_units = self.encoder_architecture[layer_num - 1]

                self.params[w_name] = theano.shared( \
                    create_weight(prev_hidden_units, hidden_units), \
                    name=w_name)

                self.params[b_name] = theano.shared(\
                    create_bias(hidden_units),\
                    name=b_name)

        # Encoder produces a diagonal Gaussian over latent space
        self.params["W_hmu"] = theano.shared(\
            create_weight(self.encoder_architecture[-1], self.n_latent),\
            name="W_hmu")

        self.params["b_hmu"] = theano.shared( \
            create_bias(self.n_latent), \
            name="b_hmu")

        self.params["W_hsigma"] = theano.shared( \
            create_weight(self.encoder_architecture[-1], self.n_latent), \
            name="W_hsigma")

        self.params["b_hsigma"] = theano.shared( \
            create_bias_logsig(self.n_latent), \
            name="b_hsigma")

        # Decoder layers
        for layer_num, hidden_units in enumerate(self.decoder_architecture):
            w_name = "W_decode_"+str(layer_num)
            b_name = "b_decode_"+str(layer_num)
            self.variational_param_name_list += [w_name, b_name]
            self.variational_param_name_to_sigma[w_name] = 1.0
            self.variational_param_name_to_sigma[b_name] = 1.0

            if layer_num == 0:

                self.params[w_name+"-mu"] = theano.shared(\
                    create_weight(self.n_latent, hidden_units), \
                    name=w_name+"-mu")

                self.params[b_name+"-mu"] = theano.shared(\
                    create_bias(hidden_units), \
                    name=b_name+"-mu")

                self.params[w_name+"-log_sigma"] = theano.shared( \
                    create_weight_logsig(self.n_latent, hidden_units),\
                    name=w_name+"-log_sigma")

                self.params[b_name+"-log_sigma"] = theano.shared(\
                    create_bias_logsig(hidden_units), \
                    name=b_name+"-log_sigma")
            else:
                prev_hidden_units = self.decoder_architecture[layer_num - 1]

                self.params[w_name+"-mu"] = theano.shared( \
                    create_weight(prev_hidden_units, hidden_units), \
                    name=w_name+"-mu")

                self.params[b_name+"-mu"] = theano.shared(
                    create_bias(hidden_units),\
                    name=b_name+"-mu")

                self.params[w_name+"-log_sigma"] = theano.shared(\
                    create_weight_logsig(prev_hidden_units, hidden_units), \
                    name=w_name+"-log_sigma")

                self.params[b_name+"-log_sigma"] = theano.shared( \
                    create_bias_logsig(hidden_units), \
                    name=b_name+"-log_sigma")

        #generalize so we don"t need any layers in the decoder
        if len(self.decoder_architecture) == 0:
            self.final_output_size = self.n_latent
        else:
            self.final_output_size = self.decoder_architecture[-1]

        if self.convolve_patterns:
            #make the convolution weights
            self.params["W_conv-mu"] = theano.shared(\
                create_weight(self.conv_decoder_size, self.alphabet_size),\
                name="W_conv-mu")

            self.params["W_conv-log_sigma"] = theano.shared(\
                create_weight_logsig(self.conv_decoder_size, \
                self.alphabet_size), \
                name="W_conv-log_sigma")

            #then make the output weights
            self.params["W_out-mu"] = theano.shared( \
                create_weight(self.final_output_size, \
                self.seq_len * self.conv_decoder_size), \
                name="W_out-mu")

            self.params["W_out-log_sigma"] = theano.shared( \
                create_weight_logsig(self.final_output_size, \
                self.seq_len * self.conv_decoder_size), \
                name="W_out-log_sigma")

            if self.output_bias:

                self.params["b_out-log_sigma"] = theano.shared( \
                    create_bias_logsig(self.seq_len * self.alphabet_size), \
                    name="b_out-log_sigma")

                self.params["b_out-mu"] = theano.shared(\
                    create_bias(self.seq_len * self.alphabet_size),\
                    name="b_out-mu")

            self.variational_param_name_list += ["W_conv"]
            self.variational_param_name_to_sigma["W_conv"] = 1.0


        else:
            #otherwise just make the output weights
            self.params["W_out-mu"] = theano.shared( \
                create_weight(self.final_output_size, \
                self.seq_len * self.alphabet_size), \
                name="W_out-mu")

            self.params["W_out-log_sigma"] = theano.shared( \
                create_weight_logsig(self.final_output_size, \
                self.seq_len * self.alphabet_size), \
                name="W_out-log_sigma")

            if self.output_bias:

                self.params["b_out-log_sigma"] = theano.shared( \
                    create_bias_logsig(self.seq_len * self.alphabet_size), \
                    name="b_out-log_sigma")

                self.params["b_out-mu"] = theano.shared( \
                    create_bias(self.seq_len * self.alphabet_size), \
                    name="b_out-mu")

        if self.output_bias:
            self.variational_param_name_list += ["W_out", "b_out"]
            self.variational_param_name_to_sigma["W_out"] = pattern_sigma
            self.variational_param_name_to_sigma["b_out"] = pattern_sigma
        else:
            self.variational_param_name_list += ["W_out"]
            self.variational_param_name_to_sigma["W_out"] = pattern_sigma

        # Subgroup sparsity parameters (Fadeout)
        if sparsity != False:
            self.params["W_out_scale-mu"] = theano.shared( \
                create_weight_zeros(self.final_output_size / self.n_patterns, \
                self.seq_len), \
                name="W_out_scale-mu")

            self.params["W_out_scale-log_sigma"] = theano.shared( \
                create_weight_logsig(self.final_output_size / self.n_patterns, \
                self.seq_len), \
                name="W_out_scale-log_sigma")

        if self.final_pwm_scale:
            self.variational_param_name_list += ["final_pwm_scale"]
            self.variational_param_name_to_sigma["final_pwm_scale"] = 1.0
            self.params["final_pwm_scale-mu"] = theano.shared( \
                np.ones(1).astype(theano.config.floatX), \
                name="final_pwm_scale-mu")

            self.params["final_pwm_scale-log_sigma"] = theano.shared( \
                -5 * np.ones(1).astype(theano.config.floatX), \
                name="final_pwm_scale-log_sigma")

        # Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key, value in self.params.items():
            self.m[key] = theano.shared( \
                np.zeros_like(value.get_value()).astype(theano.config.floatX), \
                name="m_" + key)

            self.v[key] = theano.shared( \
                np.zeros_like(value.get_value()).astype(theano.config.floatX), \
                name="v_" + key)

        # # Random number generation
        if "gpu" in theano.config.device or "cuda" in theano.config.device:
            self.srng = RandomStreams(seed=random_seed)
            #srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            self.srng = T.shared_randomstreams.RandomStreams(seed=random_seed)

        # Gradient functions
        self.update, self.encode, self.decode, \
            self.recognize, self.likelihoods, self.all_likelihood_components, \
            self.get_pattern_activations \
            = self.create_gradientfunctions()

    def KLD_diag_gaussians(self, mu, log_sigma, prior_mu, prior_log_sigma):
        """ KL divergence between two Diagonal Gaussians """
        return prior_log_sigma - log_sigma + 0.5 * (T.exp(2. * log_sigma) \
            + T.sqr(mu - prior_mu)) * T.exp(-2. * prior_log_sigma) - 0.5

    def _encode_nonlinearity(self,x):
        """Nonlinearity to be used for all layers"""
        if self.encode_nonlinearity_type == "relu":
            return T.switch(x<0, 0, x)
        elif self.encode_nonlinearity_type == "tanh":
            return T.tanh(x)
        elif self.encode_nonlinearity_type == "sigmoid":
            return T.nnet.sigmoid(x)
        elif self.encode_nonlinearity_type == "elu":
            return T.switch(x<0, (T.exp(x)-1.0), x)

    def _decode_nonlinearity(self,x):
        """Nonlinearity to be used for all layers"""
        if self.decode_nonlinearity_type == "relu":
            return T.switch(x<0, 0, x)
        elif self.decode_nonlinearity_type == "tanh":
            return T.tanh(x)
        elif self.decode_nonlinearity_type == "sigmoid":
            return T.nnet.sigmoid(x)
        elif self.decode_nonlinearity_type == "elu":
            return T.switch(x<0, (T.exp(x)-1.0), x)

    def encoder(self, x):
        """Variational approximation to the posterior of the decoder"""
        batch_size,seq_len,alphabet_size = x.shape

        if self.convolve_encoder:
            x_flat = x.reshape((batch_size*seq_len,alphabet_size))
            x_conv = self._encode_nonlinearity(\
                T.dot(x_flat, self.params["W_conv_encode"]))
            x_reshaped = x_conv.reshape( \
                (batch_size,seq_len*self.conv_encoder_size))

        else:
            x_reshaped = x.reshape((batch_size,seq_len*alphabet_size))

        for layer_num in range(len(self.encoder_architecture)):
            w_name = "W_encode_"+str(layer_num)
            b_name = "b_encode_"+str(layer_num)
            if layer_num == 0:
                layer_up_val = self._encode_nonlinearity( \
                    T.dot(x_reshaped, self.params[w_name]) \
                    + self.params[b_name].dimshuffle("x", 0))

            else:
                layer_up_val = self._encode_nonlinearity( \
                    T.dot(layer_up_val, self.params[w_name]) \
                    + self.params[b_name].dimshuffle("x", 0))

        mu = T.dot(layer_up_val, self.params["W_hmu"]) \
            + self.params["b_hmu"].dimshuffle("x", 0)
        log_sigma = T.dot(layer_up_val, self.params["W_hsigma"]) \
            + self.params["b_hsigma"].dimshuffle("x", 0)

        return mu, log_sigma

    def _sampler(self, mu, log_sigma):
        """Samples from a diagonal Gaussian for stochastic variables"""
        eps = self.srng.normal(mu.shape)
        z = mu + T.exp(log_sigma) * eps
        return z

    def decoder_sparse(self, x, layer_up_val):
        """Decodes from latent space to a conditional distribution"""
        for layer_num in range(len(self.decoder_architecture)):

            w_name = "W_decode_"+str(layer_num)
            b_name = "b_decode_"+str(layer_num)

            W_mu = self.params[w_name+"-mu"]
            W_log_sigma = self.params[w_name+"-log_sigma"]
            b_mu = self.params[b_name+"-mu"]
            b_log_sigma = self.params[b_name+"-log_sigma"]

            W = self._sampler(W_mu, W_log_sigma)
            b = self._sampler(b_mu, b_log_sigma)

            #if this is my final layer, check if I should do softmax or relu
            if (layer_num + 1) == len(self.decoder_architecture):

                if self.final_decode_nonlinearity == "sigmoid":
                    layer_up_val = T.nnet.sigmoid(T.dot(layer_up_val, W) \
                        + b.dimshuffle("x", 0))

                #otherwise to the nonlinearity that was provided before
                else:
                    layer_up_val = self._decode_nonlinearity(\
                        T.dot(layer_up_val, W) + b.dimshuffle("x", 0))

            #otherwise do other nonlinearities
            else:
                layer_up_val = self._decode_nonlinearity(\
                    T.dot(layer_up_val, W) + b.dimshuffle("x", 0))

        # Unpack the output weight matrices
        W_mu = self.params["W_out-mu"]
        W_log_sigma = self.params["W_out-log_sigma"]
        W_out = self._sampler(W_mu, W_log_sigma)

        if self.sparsity != False:
            # Scale parameters for the tiled filter banks
            W_mu = self.params["W_out_scale-mu"]
            W_log_sigma = self.params["W_out_scale-log_sigma"]
            W_scale = self._sampler(W_mu, W_log_sigma)
            W_scale = T.tile(W_scale,(self.n_patterns, 1))

            if self.sparsity == "logit":
                W_scale = T.nnet.sigmoid(W_scale.dimshuffle(0,1,"x"))
            else:
                W_scale = T.exp(W_scale.dimshuffle(0,1,"x"))

        if self.convolve_patterns:

            # Sample the convolutional weights
            W_mu = self.params["W_conv-mu"]
            W_log_sigma = self.params["W_conv-log_sigma"]
            W_conv = self._sampler(W_mu,W_log_sigma)

            # Make the pattern matrix the size of the output
            W_out = T.dot(W_out.reshape((self.final_output_size \
                * self.seq_len, self.conv_decoder_size)), W_conv)

            if self.sparsity != False:
                # Apply group sparsity
                W_out = W_out.reshape((self.final_output_size, \
                    self.seq_len, self.alphabet_size)) * W_scale

            # Reshape for multiplication with upstream decoder output
            W_out = W_out.reshape((self.final_output_size, \
                self.seq_len * self.alphabet_size))

        elif self.sparsity != False:
            W_out = W_out.reshape((self.final_output_size, \
                self.seq_len, self.alphabet_size)) * W_scale
            W_out = W_out.reshape((self.final_output_size, \
                self.seq_len * self.alphabet_size))


        if self.output_bias:

            b_mu = self.params["b_out-mu"]
            b_log_sigma = self.params["b_out-log_sigma"]
            b_out = self._sampler(b_mu, b_log_sigma)
            reconstructed_x_flat = T.dot(layer_up_val, W_out) \
                + b_out.dimshuffle("x", 0)

        else:
            reconstructed_x_flat = T.dot(layer_up_val, W_out)

        if self.final_pwm_scale:
            mu = self.params["final_pwm_scale-mu"]
            log_sigma = self.params["final_pwm_scale-log_sigma"]
            pwm_scale = self._sampler(mu,log_sigma)[0]
            reconstructed_x_flat = reconstructed_x_flat \
                * T.log(1.0+T.exp(pwm_scale))

        # Batch size is always going to be the same: just reshape it with that
        reconstructed_x_unnorm = reconstructed_x_flat.reshape(\
            (layer_up_val.shape[0], self.seq_len, self.alphabet_size))

        # Softmax over amino acids
        e_x = T.exp(reconstructed_x_unnorm \
            - reconstructed_x_unnorm.max(axis=2, keepdims=True))
        reconstructed_x = e_x / e_x.sum(axis=2, keepdims=True)

        # Numerically stable softmax using logsumexp trick
        xdev = reconstructed_x_unnorm \
            - reconstructed_x_unnorm.max(2, keepdims=True)
        log_softmax = xdev - T.log(T.sum(T.exp(xdev),axis=2,keepdims=True))
        logpxz = T.sum(T.sum((x * log_softmax), axis=-1),axis=-1)

        return reconstructed_x, logpxz, layer_up_val

    def _anneal(self, update_num):
        """ Anneal the KL if using annealing"""
        # If true, return first, else return second
        KL_scale = T.switch(update_num < self.warm_up, update_num/self.warm_up, 1.0)
        return KL_scale

    def gen_kld_params(self):
        """ Generate the KL for all the variational parameters in the decoder"""
        KLD_params = 0.0
        for key_prefix in self.variational_param_name_list:
            mu = self.params[key_prefix+"-mu"].flatten()
            log_sigma = self.params[key_prefix+"-log_sigma"].flatten()
            prior_log_sigma = T.log(\
                self.variational_param_name_to_sigma[key_prefix])
            KLD_params += T.sum(-self.KLD_diag_gaussians(mu, log_sigma, \
                0.0, prior_log_sigma))
        return KLD_params

    def gen_kld_sparsity(self,sparsity):
        """ Generate the KL for the sparsity parameters """
        if sparsity == "logit":
            # Use a continuous relaxation of a spike and slab prior
            #    with a logit normit scale distribution
            KLD_fadeout = -self.KLD_diag_gaussians( \
                self.params["W_out_scale-mu"], \
                self.params["W_out_scale-log_sigma"],\
                self.logit_mu, \
                T.log(self.logit_sigma))

        elif sparsity == "analytic":
            # Use a moment-matched Gaussian approximation to the
            #   log-space Hyperbolic Secant hyperprior of the Horseshoe
            KLD_fadeout = -self.KLD_diag_gaussians( \
                self.params["W_out_scale-mu"], \
                self.params["W_out_scale-log_sigma"], \
                T.log(self.global_scale), \
                T.log(np.pi / 2))
        else:
            # Estimate KL divergence for the sparsity
            #   scale parameters (Fadeout) by sampling
            W_scale = T.exp(self._sampler(self.params["W_out_scale-mu"],\
                self.params["W_out_scale-log_sigma"]))
            if sparsity == "horseshoe":
                # Horsehoe sparsity has Half-Cauchy hyperprior
                KLD_fadeout = (T.log(2.0) + T.log(self.global_scale) \
                    - T.log(np.pi) + T.log(W_scale) \
                    - T.log(self.global_scale*self.global_scale \
                    + W_scale * W_scale)) \
                    + (self.params["W_out_scale-log_sigma"] \
                    + 0.5 * T.log(2.0 * np.pi * np.e))

            elif sparsity == "laplacian":
                # Laplace sparsity has exponential hyperprior
                KLD_fadeout = (T.log(2.0) + T.log(self.inv_global_scale) \
                    - self.inv_global_scale * W_scale * W_scale + 2.0 \
                    * T.log(W_scale)) + (self.params["W_out_scale-log_sigma"] \
                    + 0.5 * T.log(2.0 * np.pi * np.e))

            elif sparsity == "ard":
                # Automatic Relevance Determination sparsity
                #  has Inverse-Gamma hyperprior
                KLD_fadeout = (T.log(2.0) + (self.global_scale \
                    * T.log(self.global_scale)) \
                    - T.gammaln(self.global_scale) - (self.global_scale \
                    / ((W_scale * W_scale) + epsilon)) \
                    - (2.0 * self.global_scale * T.log(W_scale))) \
                    + (self.params["W_out_scale-log_sigma"] \
                    + 0.5 * T.log(2.0 * np.pi * np.e))

        return T.sum(KLD_fadeout)

    def create_gradientfunctions(self):
        """Sets up all gradient-based update functions for optimization"""
        x = T.tensor3("x")
        Neff = T.scalar("neff")
        update_num = T.scalar("update_num")

        # Encode and reconstruct in 3 simple commands
        mu, log_sigma = self.encoder(x)

        z = self._sampler(mu, log_sigma)

        reconstructed_x, logpxz, pattern_activations = \
            self.decoder_sparse(x, z)

        KLD_latent = 0.5 * T.sum(1.0 + 2.0 * log_sigma - mu**2.0 \
            - T.exp(2.0 * log_sigma), axis=1)

        #generate the kld for the parameters
        KLD_params_all = self.gen_kld_params()

        # KLD of the fadeout to the params
        if self.sparsity != False:
            KLD_params_all += self.gen_kld_sparsity(self.sparsity)

        logpx_i = logpxz + KLD_latent

        warm_up_scale = self._anneal(update_num)

        # Scale the KL if appropriate, default is 1.0 (no scaling)
        KLD_latent_update = KLD_latent * self.kl_scale

        logpx_update = T.mean(logpxz + (warm_up_scale * KLD_latent_update)) \
            + (warm_up_scale * (KLD_params_all / Neff))

        # Compute all the gradients
        gradients = T.grad(logpx_update, self.params.values())

        # Adam implemented as updates
        updates = self.get_adam_updates(gradients)

        update = theano.function([x, Neff, update_num], \
            [logpx_update, T.mean(logpxz), (KLD_params_all/Neff), \
            T.mean(KLD_latent)], updates=updates, \
            allow_input_downcast=True)

        likelihoods = theano.function([x], logpx_i, \
            allow_input_downcast=True)

        all_likelihood_components = theano.function([x], \
            [logpx_i,KLD_latent,logpxz], \
            allow_input_downcast=True)

        encode = theano.function([x], z, \
            allow_input_downcast=True)

        decode = theano.function([z], reconstructed_x, \
            allow_input_downcast=True)

        get_pattern_activations = theano.function([x], pattern_activations, \
            allow_input_downcast=True)

        recognize = theano.function([x], [mu, log_sigma], \
            allow_input_downcast=True)

        return update, encode, decode, recognize, likelihoods,\
            all_likelihood_components, get_pattern_activations

    def save_parameters(self, file_prefix):
        """Saves all the parameters in a way they can be retrieved later"""
        cPickle.dump({name: p.get_value() for name, p in self.params.items()},\
            open(self.working_dir+"/params/"+file_prefix + "_params.pkl", "wb"))
        cPickle.dump({name: m.get_value() for name, m in self.m.items()}, \
            open(self.working_dir+"/params/"+file_prefix +"_m.pkl", "wb"))
        cPickle.dump({name: v.get_value() for name, v in self.v.items()}, \
            open(self.working_dir+"/params/"+file_prefix +"_v.pkl", "wb"))

    def load_parameters(self, file_prefix=""):
        """Load the variables in a shared variable safe way"""
        p_list = cPickle.load(open(self.working_dir+"/params/"+file_prefix \
            + "_params.pkl", "rb"))
        m_list = cPickle.load(open(self.working_dir+"/params/"+file_prefix \
            + "_m.pkl", "rb"))
        v_list = cPickle.load(open(self.working_dir+"/params/"+file_prefix \
            + "_v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

    def get_adam_updates(self, gradients):
        """Computes SGD updates for model parameters with Adam"""
        updates = OrderedDict()

        self.t = self.t + 1.

        gamma = T.sqrt(1. - self.b2**self.t) / (1. - self.b1**self.t)

        values_iterable = zip(self.params.keys(), self.params.values(),\
            gradients, self.m.values(), self.v.values())

        for name, parameter, gradient, m, v in values_iterable:

            new_m = self.b1 * m + (1. - self.b1) * gradient
            new_v = self.b2 * v + (1. - self.b2) * (gradient**2)

            updates[parameter] = parameter + self.learning_rate * gamma \
                * new_m / (T.sqrt(new_v) + epsilon)

            updates[m] = new_m
            updates[v] = new_v

        return updates

class VariationalAutoencoderMLE:
    """
    This class implements a Variational Autoencoder

    Parameters
    --------------
    data: processed data class from helper.py
    encoder_architecture: List for size of layers of encoder
    decoder_architecture: List for size of layers of decoder
    n_latent: Number of latent variables
    n_patterns: Number of times the scale parameters should be tiled over the
                    final weights of the decoder
    batch_size: Mini-batch size
    encode_nonlinearity_type: Nonlinearity of encoder
    decode_nonlinearity_type: Nonlinearity of decoder
    final_decode_nonlinearity: Final nonlinearity of decoder
    sparsity: Sparsity type, using a noncentered reparameterization.
                Options include: logit, analytic, laplacian, horseshoe, ard
                See Ingraham and Marks, 2016 (https://arxiv.org/abs/1602.03807)
    global_scale: Global scale prior for sparsity: analytic, laplacian, horseshoe, ard
    logit_p: Global scale prior for logit sparsity
    logit_sigma: Prior sigma for scale prior for logit sparsity
    sparsity_lambda: Regularization strength of sparsity parameters
    l2_lambda: Regularization strength of decoder parameters
    warm_up: Annealing schedule for KL (default 0)
    convolve_encoder: Include 1D conv on the input sequences
    convolve_patterns: Include 1D conv on the final decoder weights
                        Also known as the dictionary
    conv_encoder_size: Convolution size for input
    conv_decoder_size: Convolution size for dictionary
    output_bias: Include an output bias
    final_pwm_scale: Include inverse temperature parameter
    working_dir: directory to save and load parameters
    kl_scale: Scale of KL of latent variables, default 1.0
                Scale < 1.0 approaches a normal autoencoder
                Scale > 1.0 turns into beta-autoencoder (Higgins et al, 2016)
    learning_rate: Adam learning rate
    b1: Adam b1 hyperparameter
    b2: Adam b2 hyperparameter
    random_seed: Random init seed
    dropout: Include dropout on the decoder (Keep prob = 0.5)

    Returns
    ------------
    None (Purpose of the file is to make callables for training and inference)

    """

    def __init__(self,
            data,
            encoder_architecture=[1500,1500],
            decoder_architecture=[100,500],
            n_latent=2,
            n_patterns=4,
            batch_size=100,
            encode_nonlinearity_type="relu",
            decode_nonlinearity_type="relu",
            final_decode_nonlinearity="sigmoid",
            global_scale=1.0,
            convolve_encoder=False,
            convolve_patterns=True,
            conv_decoder_size=10,
            conv_encoder_size=10,
            warm_up=0.0,
            output_bias=True,
            final_pwm_scale=False,
            working_dir=".",
            learning_rate=0.001,
            b1=0.9,
            b2=0.999,
            random_seed=42,
            sparsity_lambda=0.0,
            l2_lambda=0.0,
            sparsity='logit',
            kl_scale=1.0,
            logit_p=0.01,
            logit_sigma=4.0,
            dropout=False):

        # Directory for saving parameters
        self.working_dir = working_dir

        # Network architecture
        self.n_latent = n_latent
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture
        self.n_patterns = n_patterns

        # For cases that go from latent -> output
        if self.decoder_architecture != []:
            self.decoder_architecture[-1] = self.decoder_architecture[-1] \
                * self.n_patterns
        else:
            self.n_latent = n_latent * self.n_patterns

        self.seq_len = data.seq_len
        self.alphabet_size = data.alphabet_size
        self.sparsity_lambda = sparsity_lambda
        self.l2_lambda = l2_lambda
        self.decode_nonlinearity_type = decode_nonlinearity_type
        self.encode_nonlinearity_type = encode_nonlinearity_type
        self.sparsity = sparsity
        self.kl_scale = theano.shared(kl_scale).astype(theano.config.floatX)
        self.global_scale = theano.shared(global_scale).astype(theano.config.floatX)
        self.inverse_global_scale = theano.shared(1.0/global_scale).astype(theano.config.floatX)
        self.dropout = dropout
        self.final_pwm_scale = final_pwm_scale
        self.warm_up = theano.shared(warm_up).astype(theano.config.floatX)

        self.logit_p = logit_p

        # Calculate the mu with the given sigma and logit p
        #    to calculate the approximate mu
        self.logit_mu = theano.shared(np.sqrt(2.0) * logit_sigma \
            * erfinv(2.0 * logit_p - 1.0)).astype(theano.config.floatX)
        self.logit_sigma = \
            theano.shared(logit_sigma).astype(theano.config.floatX)

        #algorithmic options
        self.final_decode_nonlinearity = final_decode_nonlinearity
        self.convolve_patterns = convolve_patterns
        self.conv_decoder_size = conv_decoder_size
        self.convolve_encoder = convolve_encoder
        self.conv_encoder_size = conv_encoder_size
        self.output_bias = output_bias
        self.prng = np.random.RandomState(random_seed)

        # Learning parameters
        self.batch_size = batch_size
        self.b1 = theano.shared(b1).astype(theano.config.floatX)
        self.b2 = theano.shared(b2).astype(theano.config.floatX)
        self.learning_rate = theano.shared(learning_rate).astype(theano.config.floatX)
        self.t = theano.shared(0.0).astype(theano.config.floatX)

        self.decoder_type = 'mle'

        # Parameter initialization
        sigma_init = 0.01
        logsig_init = -5
        # Glorot initialization
        create_weight = lambda dim_input, dim_output: self.prng.normal(0.0, \
            np.sqrt(2.0 / (dim_input + dim_output)), \
            (dim_input, dim_output)).astype(theano.config.floatX)
        create_weight_zeros = lambda dim_input, dim_output: \
            np.zeros((dim_input, dim_output)).astype(theano.config.floatX)
        create_bias = lambda dim_output:  0.1 \
            * np.ones(dim_output).astype(theano.config.floatX)
        # Variational uncertainty
        create_weight_logsig = lambda dim_input, dim_output: logsig_init \
            * np.ones((dim_input, dim_output)).astype(theano.config.floatX)
        create_bias_logsig = lambda dim_output: logsig_init \
            * np.ones(dim_output).astype(theano.config.floatX)

        self.params = OrderedDict()
        self.variational_param_name_list = []

        # Encoder Layers
        for layer_num, hidden_units in enumerate(self.encoder_architecture):
            w_name = 'W_encode_'+str(layer_num)
            b_name = 'b_encode_'+str(layer_num)
            if layer_num == 0:

                if self.convolve_encoder:

                    self.params['W_conv_encode'] = theano.shared(\
                        create_weight(self.alphabet_size,\
                        self.conv_encoder_size), name='W_conv_encode')

                    self.params[w_name] = theano.shared(\
                        create_weight(self.seq_len * self.conv_encoder_size, \
                        hidden_units), name=w_name)
                    self.params[b_name] = theano.shared(\
                        create_bias(hidden_units), name=b_name)

                else:
                    self.params[w_name] = theano.shared(\
                        create_weight(self.seq_len * self.alphabet_size, \
                        hidden_units), name=w_name)
                    self.params[b_name] = theano.shared(\
                        create_bias(hidden_units), name=b_name)

            else:
                prev_hidden_units = self.encoder_architecture[layer_num - 1]
                self.params[w_name] = theano.shared(\
                    create_weight(prev_hidden_units, hidden_units), name=w_name)
                self.params[b_name] = theano.shared(create_bias(hidden_units), \
                    name=b_name)

        # Encoder produces a diagonal Gaussian over latent space
        self.params['W_hmu'] = theano.shared(create_weight(\
            self.encoder_architecture[-1], self.n_latent), name='W_hmu')
        self.params['b_hmu'] = theano.shared(create_bias(self.n_latent), \
            name='b_hmu')
        self.params['W_hsigma'] = theano.shared(create_weight(\
            self.encoder_architecture[-1], self.n_latent), name='W_hsigma')
        self.params['b_hsigma'] = theano.shared(create_bias_logsig(self.n_latent), \
            name='b_hsigma')

        # Decoder layers
        for layer_num, hidden_units in enumerate(self.decoder_architecture):
            w_name = 'W_decode_'+str(layer_num)
            b_name = 'b_decode_'+str(layer_num)
            if layer_num == 0:
                self.params[w_name] = theano.shared(create_weight(\
                    self.n_latent, hidden_units), name=w_name)
                self.params[b_name] = theano.shared(create_bias(hidden_units), \
                    name=b_name)

            else:
                prev_hidden_units = self.decoder_architecture[layer_num - 1]
                self.params[w_name] = theano.shared(create_weight(\
                    prev_hidden_units, hidden_units), name=w_name)
                self.params[b_name] = theano.shared(create_bias(hidden_units), \
                    name=b_name)


        #generalize so we don't need any layers in the decoder
        if len(self.decoder_architecture) == 0:
            self.final_output_size = self.n_latent
        else:
            self.final_output_size = self.decoder_architecture[-1]

        # Output layers (logos)
        if self.convolve_patterns:
            #make the convolution weights
            self.params['W_conv_decode'] = theano.shared(create_weight(\
                self.conv_decoder_size, self.alphabet_size), name='W_conv')

            #then make the output weights
            self.params['W_out'] = theano.shared(create_weight(\
                self.final_output_size, self.seq_len * self.conv_decoder_size),\
                name='W_out')

            if self.output_bias:
                self.params['b_out'] = theano.shared(create_bias(self.seq_len \
                * self.alphabet_size), name='b_out')

        else:
            #otherwise just make the output weights
            self.params['W_out'] = theano.shared(create_weight(\
                self.final_output_size, self.seq_len * self.alphabet_size), \
                name='W_out')
            if self.output_bias:
                self.params['b_out'] = theano.shared(create_bias(self.seq_len \
                    * self.alphabet_size), name='b_out')

        if self.sparsity != False:
            self.params['W_out_scale'] = theano.shared(create_weight(\
                self.final_output_size / self.n_patterns, self.seq_len), \
                name='W_out_scale')

        if self.final_pwm_scale:
            self.params['final_pwm_scale'] = theano.shared(np.ones(1).astype(\
                theano.config.floatX), name='final_pwm_scale')

        # Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()
        for key, value in self.params.items():
            self.m[key] = theano.shared(np.zeros_like(\
                value.get_value()).astype(theano.config.floatX), name='m_' + key)
            self.v[key] = theano.shared(np.zeros_like(\
                value.get_value()).astype(theano.config.floatX), name='v_' + key)

        # Random number generation
        if "gpu" in theano.config.device or "cuda" in theano.config.device:
            self.srng = RandomStreams(seed=random_seed)
            #srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            self.srng = T.shared_randomstreams.RandomStreams(seed=random_seed)

        # Gradient functions
        self.update, self.encode, self.decode, self.recognize, \
            self.likelihoods, self.all_likelihood_components, \
            self.get_pattern_activations = self.create_gradientfunctions()


    def encoder(self, x):
        """Variational approximation to the posterior of the decoder"""
        batch_size,seq_len,alphabet_size = x.shape

        if self.convolve_encoder:
            x_flat = x.reshape((batch_size*seq_len,alphabet_size))
            x_conv = self.encode_nonlinearity(T.dot(x_flat, \
                self.params['W_conv_encode']))
            x_reshaped = x_conv.reshape((batch_size,\
                seq_len*self.conv_encoder_size))

        else:
            x_reshaped = x.reshape((batch_size,seq_len*alphabet_size))

        for layer_num in range(len(self.encoder_architecture)):
            w_name = 'W_encode_'+str(layer_num)
            b_name = 'b_encode_'+str(layer_num)
            if layer_num == 0:
                layer_up_val = self._encode_nonlinearity(T.dot(x_reshaped, \
                    self.params[w_name])+self.params[b_name].dimshuffle('x', 0))
            else:
                layer_up_val = self._encode_nonlinearity(T.dot(layer_up_val, \
                    self.params[w_name])+self.params[b_name].dimshuffle('x', 0))

        mu = T.dot(layer_up_val, self.params['W_hmu']) \
            + self.params['b_hmu'].dimshuffle('x', 0)
        log_sigma = T.dot(layer_up_val, self.params['W_hsigma']) \
            + self.params['b_hsigma'].dimshuffle('x', 0)

        return mu, log_sigma

    def _sampler(self, mu, log_sigma):
        """Samples from a diagonal Gaussian for stochastic variables"""
        eps = self.srng.normal(mu.shape)
        z = mu + T.exp(log_sigma) * eps
        return z

    def _encode_nonlinearity(self, x):
        """Nonlinearity to be used for all layers"""
        if self.encode_nonlinearity_type == 'relu':
            return T.switch(x<0, 0, x)
        elif self.encode_nonlinearity_type == 'tanh':
            return T.tanh(x)
        elif self.encode_nonlinearity_type == 'sigmoid':
            return T.nnet.sigmoid(x)
        elif self.encode_nonlinearity_type == 'elu':
            return T.switch(x<0, (T.exp(x)-1.0), x)

    def _decode_nonlinearity(self, x):
        """Nonlinearity to be used for all layers"""
        if self.decode_nonlinearity_type == 'relu':
            return T.switch(x<0, 0, x)
        elif self.decode_nonlinearity_type == 'tanh':
            return T.tanh(x)
        elif self.decode_nonlinearity_type == 'sigmoid':
            return T.nnet.sigmoid(x)
        elif self.decode_nonlinearity_type == 'elu':
            return T.switch(x<0, (T.exp(x)-1.0), x)

    def _final_decode_nonlinearity(self,x):
        if self.final_decode_nonlinearity == 'relu':
            return T.switch(x<0, 0, x)
        elif self.final_decode_nonlinearity == 'tanh':
            return T.tanh(x)
        elif self.final_decode_nonlinearity == 'sigmoid':
            return T.nnet.sigmoid(x)
        elif self.final_decode_nonlinearity == 'elu':
            return T.switch(x<0, (T.exp(x)-1.0), x)

    def _dropout(self, x, rescale=True, dropout_p=0.5):

        retain_prob = 1.0 - dropout_p

        if rescale:
            x /= retain_prob

        dropout_mask = self.srng.binomial(x.shape, p=retain_prob, dtype=x.dtype)

        return x * dropout_mask

    def decoder(self, x, layer_up_val):
        """Decodes from latent space to a conditional distribution"""

        for layer_num in range(len(self.decoder_architecture)):
            w_name = 'W_decode_'+str(layer_num)
            b_name = 'b_decode_'+str(layer_num)
            W = self.params[w_name]
            b = self.params[b_name]

            #if this is my final layer, check if I should do softmax or relu
            if (layer_num + 1) == len(self.decoder_architecture):

                layer_up_val = self._final_decode_nonlinearity(\
                    T.dot(layer_up_val, W) + b.dimshuffle('x', 0))

                if self.dropout:
                    layer_up_val = self._dropout(layer_up_val)

            #otherwise do other nonlinearities
            else:
                layer_up_val = self._decode_nonlinearity(T.dot(layer_up_val, W)\
                    + b.dimshuffle('x', 0))

                if self.dropout:
                    layer_up_val = self._dropout(layer_up_val)

        W_out = self.params['W_out']
        b_out = self.params['b_out']

        if self.convolve_patterns:

            # Sample the convolutional weights
            W_conv = self.params['W_conv_decode']

            # Make the pattern matrix the size of the output
            W_out = T.dot(W_out.reshape((self.final_output_size \
                * self.seq_len, self.conv_decoder_size)), W_conv)

            # Apply group sparsity
            W_out = W_out.reshape((self.final_output_size, self.seq_len,\
                self.alphabet_size))

            # Reshape for multiplication with upstream decoder output
            W_out = W_out.reshape((self.final_output_size, self.seq_len \
                * self.alphabet_size))

        if self.sparsity != False:

            W_scale = T.tile(self.params['W_out_scale'], (self.n_patterns, 1))

            if self.sparsity == 'logit':
                W_out = W_out.reshape((self.final_output_size, self.seq_len,\
                    self.alphabet_size)) \
                    * T.nnet.sigmoid(W_scale.dimshuffle(0,1,'x'))
            else:
                W_out = W_out.reshape((self.final_output_size, self.seq_len,\
                    self.alphabet_size)) * T.exp(W_scale.dimshuffle(0,1,'x'))

            W_out = W_out.reshape((self.final_output_size, self.seq_len \
                * self.alphabet_size))


        if self.output_bias:
            reconstructed_x_flat = T.dot(layer_up_val, W_out) \
                + b_out.dimshuffle('x', 0)

        else:
            reconstructed_x_flat = T.dot(layer_up_val, W_out)

        #the batch size is always going to be the same, so just reshape it with that
        reconstructed_x_unnorm = reconstructed_x_flat.reshape(\
            (layer_up_val.shape[0], self.seq_len, self.alphabet_size))

        if self.final_pwm_scale:
            reconstructed_x_unnorm = reconstructed_x_unnorm \
                * T.log(1.0+T.exp(self.params['final_pwm_scale'][0]))

        # Softmax over amino acids
        e_x = T.exp(reconstructed_x_unnorm \
            - reconstructed_x_unnorm.max(axis=2, keepdims=True))
        reconstructed_x = e_x / e_x.sum(axis=2, keepdims=True)

        # Logsumexp softmax trick
        xdev = reconstructed_x_unnorm\
            -reconstructed_x_unnorm.max(2,keepdims=True)
        log_softmax = xdev - T.log(T.sum(T.exp(xdev),axis=2,keepdims=True))
        logpxz = T.sum(T.sum((x * log_softmax), axis=-1),axis=-1)

        return reconstructed_x, logpxz, layer_up_val

    def _anneal(self, update_num):
        # If true, return first, else return second
        KL_scale = T.switch(update_num < self.warm_up, update_num/self.warm_up, 1.0)
        return KL_scale

    def create_gradientfunctions(self):
        """Sets up all gradient-based update functions for optimization"""
        x = T.tensor3("x")
        Neff = T.scalar("Neff")
        update_num = T.scalar("update_num")

        batch_size = x.shape[0]

        # Encode and reconstruct in 3 simple commands
        mu, log_sigma = self.encoder(x)

        z = self._sampler(mu, log_sigma)
        reconstructed_x, logpxz, pattern_activations = self.decoder(x,z)

        # Negative KLD for the latent dimension
        KLD_latent = 0.5 * T.sum(1.0 + 2.0 * log_sigma - mu**2.0 \
            - T.exp(2.0 * log_sigma), axis=1)

        if self.sparsity != False:
            l2_loss = 0.0
            if self.l2_lambda > 0.0:
                for weight_name in self.params.keys():
                    if 'decode' in weight_name:
                        l2_loss += 0.5*T.sum(self.params[weight_name]\
                            *self.params[weight_name])

                l2_loss += 0.5*T.sum(self.params['b_out']*self.params['b_out'])
                l2_loss += 0.5*T.sum(self.params['W_out']*self.params['W_out'])

                if self.final_pwm_scale:
                    l2_loss += 0.5*T.sum(self.params['final_pwm_scale']\
                        *self.params['final_pwm_scale'])

            if self.sparsity == "logit":
                # Use a continuous relaxation of a spike and slab prior
                #    with a logit normit scale distribution
                group_sparsity_loss = - T.sum((-0.5*T.log(2.*np.pi\
                    *self.logit_sigma**2.))
                    - ((self.params['W_out_scale']-self.logit_mu)**2.\
                        /(2.*(self.logit_sigma**2.))))

            elif self.sparsity == "analytic":
                # Use a moment-matched Gaussian approximation to the
                #   log-space Hyperbolic Secant hyperprior of the Horseshoe
                analytic_mu = T.log(self.global_scale)
                analytic_sigma = np.pi / 2.

                group_sparsity_loss = - T.sum((-0.5*T.log(2.*np.pi\
                    *(analytic_sigma**2.)))
                    - ((self.params['W_out_scale']-analytic_mu)**2.\
                        /(2.*(analytic_sigma**2.))))

            else:
                # Estimate KL divergence for the sparsity
                #   scale parameters (Fadeout) by sampling

                W_out_scale_exp = T.exp(self.params['W_out_scale'])
                if self.sparsity == "horseshoe":
                    # Horsehoe sparsity has Half-Cauchy hyperprior
                    group_sparsity_loss = -T.sum(T.log(2.0) + T.log(self.global_scale) \
                        - T.log(np.pi) + T.log(W_out_scale_exp) \
                        - T.log(self.global_scale*self.global_scale \
                        + W_out_scale_exp * W_out_scale_exp))

                elif self.sparsity == "laplacian":
                    # Laplace sparsity has exponential hyperprior
                    group_sparsity_loss = -T.sum(T.log(2.0)\
                        + T.log(self.inverse_global_scale)\
                        - self.inverse_global_scale * W_out_scale_exp\
                        * W_out_scale_exp + 2.0 * self.params['W_out_scale'])

                elif self.sparsity == "ard":
                    # Automatic Relevance Determination sparsity
                    #  has Inverse-Gamma hyperprior
                    group_sparsity_loss = -T.sum(T.log(2.0) + (self.global_scale \
                        * T.log(self.global_scale)) \
                        - T.gammaln(self.global_scale) - (self.global_scale \
                        / ((W_out_scale_exp * W_out_scale_exp) + epsilon)) \
                        - (2.0 * self.global_scale * T.log(W_out_scale_exp)))

        else:
            l2_loss = 0.0
            if self.l2_lambda > 0.0:
                for weight_name in self.params.keys():
                    if 'decode' in weight_name:
                        l2_loss += T.sum(self.params[weight_name]\
                            *self.params[weight_name])

                l2_loss += T.sum(self.params['b_out']*self.params['b_out'])

                if self.final_pwm_scale:
                    l2_loss += T.sum(self.params['final_pwm_scale']\
                        *self.params['final_pwm_scale'])

            #first take the l2 regularization of the groups
            group_sparsity_loss = 0.0
            if self.sparsity_lambda > 0.0:

                W_out_lasso = self.params['W_out'].reshape(\
                    (self.final_output_size, self.seq_len, self.alphabet_size))
                group_sparsity_loss = T.sum(T.sqrt(T.sum(W_out_lasso \
                    * W_out_lasso,axis=2)+epsilon))

        logpx_i = logpxz + KLD_latent

        regularziation_loss = (-(self.sparsity_lambda * group_sparsity_loss)\
            - (self.l2_lambda * l2_loss)) / Neff

        warm_up_scale = self._anneal(update_num)

        # Scale the KL if appropriate, default is 1.0 (no scaling)
        KLD_latent_update = KLD_latent * self.kl_scale

        # If using importance sampling for the data, no weights are included
        logpx_update = T.mean(logpxz + warm_up_scale * KLD_latent_update) \
            + (warm_up_scale * regularziation_loss)

        # Compute all the gradients
        gradients = T.grad(logpx_update, self.params.values())

        # Adam implemented as updates
        updates = self.get_adam_updates(gradients)

        # Compile functions - this is where all the runtime cost comes from

        update = theano.function([x, Neff, update_num], \
            [logpx_update, T.mean(logpxz), (regularziation_loss), \
            T.mean(KLD_latent)], updates=updates, \
            allow_input_downcast=True)

        likelihoods = theano.function([x], logpx_i, \
            allow_input_downcast=True)

        all_likelihood_components = theano.function([x], \
            [logpx_i,KLD_latent,logpxz], \
            allow_input_downcast=True)

        encode = theano.function([x], z, \
            allow_input_downcast=True)

        decode = theano.function([z], reconstructed_x, \
            allow_input_downcast=True)

        get_pattern_activations = theano.function([x], pattern_activations, \
            allow_input_downcast=True)

        recognize = theano.function([x], [mu, log_sigma], \
            allow_input_downcast=True)

        return update, encode, decode, recognize, likelihoods,\
            all_likelihood_components, get_pattern_activations


    def save_parameters(self, file_prefix):
        """Saves all the parameters in a way they can be retrieved later"""
        cPickle.dump({name: p.get_value() for name, p in self.params.items()},\
            open(self.working_dir+"/params/"+file_prefix + "_params.pkl", "wb"))
        cPickle.dump({name: m.get_value() for name, m in self.m.items()}, \
            open(self.working_dir+"/params/"+file_prefix +"_m.pkl", "wb"))
        cPickle.dump({name: v.get_value() for name, v in self.v.items()}, \
            open(self.working_dir+"/params/"+file_prefix +"_v.pkl", "wb"))

    def load_parameters(self, file_prefix=""):
        """Load the variables in a shared variable safe way"""
        p_list = cPickle.load(open(self.working_dir+"/params/"+file_prefix \
            + "_params.pkl", "rb"))
        m_list = cPickle.load(open(self.working_dir+"/params/"+file_prefix \
            + "_m.pkl", "rb"))
        v_list = cPickle.load(open(self.working_dir+"/params/"+file_prefix \
            + "_v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

    def get_adam_updates(self, gradients):
        """Computes SGD updates for model parameters with Adam"""
        updates = OrderedDict()

        self.t = self.t + 1.

        gamma = T.sqrt(1. - self.b2**self.t) / (1. - self.b1**self.t)

        values_iterable = zip(self.params.keys(), self.params.values(),\
            gradients, self.m.values(), self.v.values())

        for name, parameter, gradient, m, v in values_iterable:

            new_m = self.b1 * m + (1. - self.b1) * gradient
            new_v = self.b2 * v + (1. - self.b2) * (gradient**2)

            updates[parameter] = parameter + self.learning_rate * gamma \
                * new_m / (T.sqrt(new_v) + epsilon)

            updates[m] = new_m
            updates[v] = new_v

        return updates
