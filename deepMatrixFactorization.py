import keras
from keras.models import Model

from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Multiply, Add
from keras.layers import Concatenate
from keras.layers import Lambda
from keras import backend
from keras import optimizers

from keras import regularizers
from keras.utils import plot_model

import matplotlib.pyplot as plt

import copy
import os
from datetime import datetime

class KerasBase:
    def __init__(self):
        return

    @staticmethod
    def make_output_dir(base_dir_name=os.path.join('.','result'), dir_name='result', with_datetime=True):
        '''
        make output directory
        return output direcotry path
        '''
        dir_path = dir_name
        if with_datetime:
            dir_path = dir_path + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dir_path = os.path.join(base_dir_name, dir_path)

        #
        os.makedirs(dir_path, exist_ok=True)

        return dir_path

    @staticmethod
    def model_visualize(model, save_filename, show_shapes=True, show_layer_names=True):
        # https://keras.io/ja/visualization/
        plot_model(model, to_file=save_filename, show_shapes=show_shapes, show_layer_names=show_layer_names)
        return

    @staticmethod
    def model_summary(model, save_filename=None, print_console=True):
        '''
        save model summary to *.txt.
        print model summary to console.
        '''
        # save model to txt
        if save_filename is not None:
            with open(save_filename, "w") as fp:
                model.summary(print_fn=lambda x: fp.write(x + "\n"))

        #
        if print_console:
            model.summary()

        return

    @staticmethod
    def save_learning_history_acc_loss(histroy, val=True, filename=None, show=False):
        '''
        history : return of model.fit()
        '''
        fig = plt.figure()

        #
        epochs = range(len(histroy.history['acc']))

        # acc
        ax_acc = fig.add_subplot(2, 1, 1)
        ax_acc.set_title('accuracy')
        # train
        label = 'acc'
        ax_acc.plot(epochs, histroy.history[label], label=label)
        if val:
            # validation
            label = 'val_acc'
            ax_acc.plot(epochs, histroy.history[label], label=label)
        ax_acc.legend()

        # loss
        ax_loss = fig.add_subplot(2, 2, 1)
        ax_loss.set_title('loss')
        # train
        label = 'loss'
        ax_loss.plot(epochs, histroy.history[label], label=label)
        if val:
            # validation
            label = 'val_loss'
            ax_loss.plot(epochs, histroy.history[label], label=label)
        ax_loss.legend()

        # save figure
        if filename is not None:
            fig.savefig(filename)

        # show
        if show:
            fig.show()

        return

    @staticmethod
    def save_learning_history_loss(histroy, val=True, filename=None, show=False):
        '''
        history : return of model.fit()
        '''
        fig = plt.figure()

        #
        epochs = range(len(histroy.history['loss']))

        # loss
        ax_loss = fig.add_subplot(1, 1, 1)
        ax_loss.set_title('loss')
        # train
        label = 'loss'
        ax_loss.plot(epochs, histroy.history[label], label=label)
        if val:
            # validation
            label = 'val_loss'
            ax_loss.plot(epochs, histroy.history[label], label=label)
        ax_loss.legend()

        # save figure
        if filename is not None:
            fig.savefig(filename)

        # show
        if show:
            fig.show()

        return


    @staticmethod
    def activation(act='relu'):
        if act == 'relu':
            return Activation('relu')
        elif act == 'lrelu':
            return LeakyReLU()
        elif act == 'linear':
            return Activation('linear')
        else:
            return Activation(act)

        return

    @staticmethod
    def sum_layer(name=None):
        def func_sum(x_):
            return keras.backend.sum(x_, axis=-1, keepdims=True)
        return Lambda(func_sum, output_shape=(1,), name=name)

    @staticmethod
    def mean_layer(name=None):
        def func_mean(x_):
            return keras.backend.mean(x_, axis=1, keepdims=True)
        return Lambda(func_mean, output_shape=(1,), name=name)

class DeepMatrixFactorization:
    def __init__(self, unique_user_num, unique_item_num, all_rating_mean=0, rating_scale=1):

        self.unique_user_num = unique_user_num
        self.unique_item_num = unique_item_num
        self.all_rating_mean = all_rating_mean
        self.rating_scale = rating_scale

        self.model = None
        self.history = None

        self.__count_call_sum_model = 0
        self.__count_call_mean_model = 0

        return

    #make model
    def make_model_mf(self, user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=0):
        '''
        make normal matrix factorization model with keras.
        rating = all_mean + user_bias + item_bias + cross_term
        '''
        input_user_id = Input(shape=(1,), name='user_id')
        input_item_id = Input(shape=(1,), name='item_id')

        #user bias
        u_bias = None
        if user_bias:
            u_bias = self.__bias_term(input_id=input_user_id, unique_id_num=self.unique_user_num, l2=0)

        #item bias
        i_bias = None
        if item_bias:
            i_bias = self.__bias_term(input_id=input_item_id, unique_id_num=self.unique_item_num, l2=0)

        #cross term
        crs_trm = None
        if cross_term:
            crs_u = self.__single_term(input_id=input_user_id, unique_id_num=self.unique_user_num, output_dim=latent_num, l2=cross_term_l2)
            crs_i = self.__single_term(input_id=input_item_id, unique_id_num=self.unique_item_num, output_dim=latent_num, l2=cross_term_l2)
            crs_trm = self.__cross_term(crs_u, crs_i, merge='sum')

        #concatenate
        def append_isNotNone(lst, v):
            tls = copy.copy(lst)
            if v is not None:
                tls.append(v)
            return tls
        concats = []
        concats = append_isNotNone(concats, u_bias)
        concats = append_isNotNone(concats, i_bias)
        concats = append_isNotNone(concats, crs_trm)
        
        if len(concats) > 1:
            y = Add(name='add_bias_crossTerm')(concats)
        else:
            y = concats[0]

        # add mean
        y = Lambda(lambda x: x*self.rating_scale + self.all_rating_mean, name='scaling')(y)

        self.model = Model(inputs=[input_user_id, input_item_id], outputs=y)

        return

    def make_model_dmf_deepLatent(self, user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=0, hidden_nodes_latent=[10], hidden_l2=[0], hidden_dropout_rates=[]):
        '''
        make normal matrix factorization model with keras.
        rating = all_mean + user_bias + item_bias + cross_term
        '''
        input_user_id = Input(shape=(1,), name='user_id')
        input_item_id = Input(shape=(1,), name='item_id')

        #user bias
        u_bias = None
        if user_bias:
            u_bias = self.__bias_term(input_id=input_user_id, unique_id_num=self.unique_user_num, l2=0)

        #item bias
        i_bias = None
        if item_bias:
            i_bias = self.__bias_term(input_id=input_item_id, unique_id_num=self.unique_item_num, l2=0)

        #cross term
        crs_trm = None
        if cross_term:
            crs_u = self.__single_term(input_id=input_user_id, unique_id_num=self.unique_user_num, output_dim=latent_num, l2=cross_term_l2, hidden_nodes=hidden_nodes_latent, hidden_l2s=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)
            crs_i = self.__single_term(input_id=input_item_id, unique_id_num=self.unique_item_num, output_dim=latent_num, l2=cross_term_l2, hidden_nodes=hidden_nodes_latent, hidden_l2s=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)
            crs_trm = self.__cross_term(crs_u, crs_i, merge='sum')

        #concatenate
        def append_isNotNone(lst, v):
            tls = copy.copy(lst)
            if v is not None:
                tls.append(v)
            return tls
        concats = []
        concats = append_isNotNone(concats, u_bias)
        concats = append_isNotNone(concats, i_bias)
        concats = append_isNotNone(concats, crs_trm)
        
        if len(concats) > 1:
            y = Add(name='add_bias_crossTerm')(concats)
        else:
            y = concats[0]

        # add mean
        y = Lambda(lambda x: x*self.rating_scale + self.all_rating_mean, name='scaling')(y)

        self.model = Model(inputs=[input_user_id, input_item_id], outputs=y)

        return

    def make_model_dmf_deepCrossterm(self, user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=0, hidden_nodes_crossterm=[10], hidden_l2=[0], hidden_dropout_rates=[]):
        '''
        make normal matrix factorization model with keras.
        rating = all_mean + user_bias + item_bias + cross_term
        '''
        input_user_id = Input(shape=(1,), name='user_id')
        input_item_id = Input(shape=(1,), name='item_id')

        #user bias
        u_bias = None
        if user_bias:
            u_bias = self.__bias_term(input_id=input_user_id, unique_id_num=self.unique_user_num, l2=0)

        #item bias
        i_bias = None
        if item_bias:
            i_bias = self.__bias_term(input_id=input_item_id, unique_id_num=self.unique_item_num, l2=0)

        #cross term
        crs_trm = None
        if cross_term:
            crs_u = self.__single_term(input_id=input_user_id, unique_id_num=self.unique_user_num, output_dim=latent_num, l2=cross_term_l2)
            crs_i = self.__single_term(input_id=input_item_id, unique_id_num=self.unique_item_num, output_dim=latent_num, l2=cross_term_l2)
            crs_trm = self.__cross_term(crs_u, crs_i, merge='sum', hidden_nodes=hidden_nodes_crossterm, hidden_l2s=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)

        #concatenate
        def append_isNotNone(lst, v):
            tls = copy.copy(lst)
            if v is not None:
                tls.append(v)
            return tls
        concats = []
        concats = append_isNotNone(concats, u_bias)
        concats = append_isNotNone(concats, i_bias)
        concats = append_isNotNone(concats, crs_trm)
        
        if len(concats) > 1:
            y = Add(name='add_bias_crossTerm')(concats)
        else:
            y = concats[0]

        # add mean
        y = Lambda(lambda x: x*self.rating_scale + self.all_rating_mean, name='scaling')(y)

        self.model = Model(inputs=[input_user_id, input_item_id], outputs=y)

        return

    def make_model_dmf_deepLatent_deepCrossterm(self, user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=0, hidden_nodes_latent=[10], hidden_nodes_crossterm=[10], hidden_l2=[0], hidden_dropout_rates=[]):
        '''
        make normal matrix factorization model with keras.
        rating = all_mean + user_bias + item_bias + cross_term
        '''
        input_user_id = Input(shape=(1,), name='user_id')
        input_item_id = Input(shape=(1,), name='item_id')

        #user bias
        u_bias = None
        if user_bias:
            u_bias = self.__bias_term(input_id=input_user_id, unique_id_num=self.unique_user_num, l2=0)

        #item bias
        i_bias = None
        if item_bias:
            i_bias = self.__bias_term(input_id=input_item_id, unique_id_num=self.unique_item_num, l2=0)

        #cross term
        crs_trm = None
        if cross_term:
            crs_u = self.__single_term(input_id=input_user_id, unique_id_num=self.unique_user_num, output_dim=latent_num, l2=cross_term_l2, hidden_nodes=hidden_nodes_latent, hidden_dropout_rates=hidden_dropout_rates)
            crs_i = self.__single_term(input_id=input_item_id, unique_id_num=self.unique_item_num, output_dim=latent_num, l2=cross_term_l2, hidden_nodes=hidden_nodes_latent, hidden_dropout_rates=hidden_dropout_rates)
            crs_trm = self.__cross_term(crs_u, crs_i, merge='sum', hidden_nodes=hidden_nodes_crossterm, hidden_l2s=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)

        #concatenate
        def append_isNotNone(lst, v):
            tls = copy.copy(lst)
            if v is not None:
                tls.append(v)
            return tls
        concats = []
        concats = append_isNotNone(concats, u_bias)
        concats = append_isNotNone(concats, i_bias)
        concats = append_isNotNone(concats, crs_trm)
        
        if len(concats) > 1:
            y = Add(name='add_bias_crossTerm')(concats)
        else:
            y = concats[0]

        # add mean
        y = Lambda(lambda x: x*self.rating_scale + self.all_rating_mean, name='scaling')(y)

        self.model = Model(inputs=[input_user_id, input_item_id], outputs=y)

        return

    def make_model_dmf_residualDeepCrossterm(self, user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=0, hidden_nodes_crossterm=[10], hidden_l2=[0], hidden_dropout_rates=[]):
        '''
        make normal matrix factorization model with keras.
        rating = all_mean + user_bias + item_bias + cross_term
        '''
        input_user_id = Input(shape=(1,), name='user_id')
        input_item_id = Input(shape=(1,), name='item_id')

        #user bias
        u_bias = None
        if user_bias:
            u_bias = self.__bias_term(input_id=input_user_id, unique_id_num=self.unique_user_num, l2=0)

        #item bias
        i_bias = None
        if item_bias:
            i_bias = self.__bias_term(input_id=input_item_id, unique_id_num=self.unique_item_num, l2=0)

        #cross term
        crs_trm = None
        res_crs_trm = None
        if cross_term:
            crs_u = self.__single_term(input_id=input_user_id, unique_id_num=self.unique_user_num, output_dim=latent_num, l2=cross_term_l2)
            crs_i = self.__single_term(input_id=input_item_id, unique_id_num=self.unique_item_num, output_dim=latent_num, l2=cross_term_l2)
            res_crs_trm = self.__res_cross_term(crs_u, crs_i, merge='sum', hidden_nodes=hidden_nodes_crossterm, hidden_l2s=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)

        #concatenate
        def append_isNotNone(lst, v):
            tls = copy.copy(lst)
            if v is not None:
                tls.append(v)
            return tls
        concats = []
        concats = append_isNotNone(concats, u_bias)
        concats = append_isNotNone(concats, i_bias)
        concats = append_isNotNone(concats, res_crs_trm)
        
        if len(concats) > 1:
            y = Add(name='add_bias_crossTerm')(concats)
        else:
            y = concats[0]

        # add mean
        y = Lambda(lambda x: x*self.rating_scale + self.all_rating_mean, name='scaling')(y)

        self.model = Model(inputs=[input_user_id, input_item_id], outputs=y)

        return


    #model of bias term
    def __single_term(self, input_id, unique_id_num, output_dim=1, 
                  hidden_nodes=[], activation='lrelu', activation_last='linear', 
                  l2=0, hidden_l2s=[], 
                  dropout_rate=0, hidden_dropout_rates=[], 
                  latent_layer_name=None):
        '''
        input -> embedding -> flatten -> dropout
            -> hidden_layer
                (-> dense -> activation -> dropout -> ... -> dense -> activation -> dropout
                 -> dense -> activation_last -> dropout)
            -> output
        '''
        #
        hidden_nodes_ = copy.copy(hidden_nodes)
        hidden_nodes_.append(output_dim)
        #
        hl = input_id
        #
        for ih, h_dim in enumerate(hidden_nodes_):
            # first layer
            if ih == 0:
                # embedding layer
                # input_shape = [batch_size, unique_id_num+1]
                # output_shape = [batch_size, input_length, output_dim]
                hl = Embedding(input_dim=unique_id_num, 
                              output_dim=hidden_nodes_[0], 
                              #input_length=1, 
                              embeddings_regularizer=regularizers.l2(l2),
                              name=latent_layer_name
                              )(hl)
                # flatten
                hl = Flatten()(hl)
                #dropout
                hl = Dropout(dropout_rate)(hl)
            
            # 2~ layer
            else:
                l2_h = 0 if len(hidden_l2s)==0 else hidden_l2s[ih-1]
                # hidden layer
                hl = Dense(h_dim, kernel_regularizer=regularizers.l2(l2_h))(hl)
                #activation
                act = activation if ih != len(hidden_nodes_)-1 else activation_last
                hl = KerasBase.activation(act)(hl)
                #dropout
                drp_rt = 0 if len(hidden_dropout_rates)==0 else hidden_dropout_rates[ih-1]
                hl = Dropout(drp_rt)(hl)

        return hl
    
    def __bias_term(self, input_id, unique_id_num, l2=0, latent_layer_name=None):
        '''
        input -> embedding -> flatten
            -> output
        '''

        bias = self.__single_term(input_id=input_id, unique_id_num=unique_id_num, output_dim=1, 
                  hidden_nodes=[], activation='lrelu', activation_last='linear', 
                  l2=l2, hidden_l2s=[], 
                  dropout_rate=0, hidden_dropout_rates=[], 
                  latent_layer_name=latent_layer_name)

        return bias

    #model of cross term
    def __cross_term(self, input1, input2, merge='sum', 
                   hidden_nodes=[], activation='lrelu', activation_last='lrelu', 
                   hidden_l2s=[], 
                   dropout_rate=0, hidden_dropout_rates=[]):
        '''
        input1 and input2 must be already embedded.
        
        (input1, input2) -> Multiply -> dropout
            -> hidden_layer
                (-> dense -> activation -> dropout -> ... -> dense -> activation -> dropout
                 -> dense -> activation_last -> dropout)
            -> merge(ex. sum, mean)
            -> output
        '''
        multiplied = Multiply()([input1, input2])
        
        #hidden layer
        hl = multiplied
        for ih, h_dim in enumerate(hidden_nodes):
            l2_h = 0 if len(hidden_l2s)==0 else hidden_l2s[ih]
            # dense
            hl = Dense(h_dim, kernel_regularizer=regularizers.l2(l2_h))(hl)
            # activation
            act = activation if ih != len(hidden_nodes)-1 else activation_last
            hl = KerasBase.activation(act)(hl)
            # dropout
            drp_rt = 0 if len(hidden_dropout_rates)==0 else hidden_dropout_rates[ih]
            hl = Dropout(drp_rt)(hl)
        
        #merge layer
        if merge=='sum':
            self.__count_call_sum_model += 1
            crs_trm = KerasBase.sum_layer(name='sum' + str(self.__count_call_sum_model))(hl)
        elif merge=='mean':
            self.__count_call_mean_model += 1
            crs_trm = KerasBase.mean_layer(name='mean' + str(self.__count_call_mean_model))(hl)

        return crs_trm

    def __res_cross_term(self, input1, input2, merge='sum', 
                   hidden_nodes=[], activation='lrelu', activation_last='lrelu', 
                   hidden_l2s=[], 
                   dropout_rate=0, hidden_dropout_rates=[]):
        '''
        input1 and input2 must be already embedded.
        
        (input1, input2) -> Multiply -> dropout
            -> hidden_layer
                (-> dense -> activation -> dropout -> ... -> dense -> activation -> dropout
                 -> dense -> activation_last -> dropout)
            -> merge(ex. sum, mean)
            -> output
        '''
        multiplied = Multiply()([input1, input2])
        
        #hidden layer
        hl = multiplied
        for ih, h_dim in enumerate(hidden_nodes):
            l2_h = 0 if len(hidden_l2s)==0 else hidden_l2s[ih]
            # dense
            hl = Dense(h_dim, kernel_regularizer=regularizers.l2(l2_h))(hl)
            # activation
            act = activation if ih != len(hidden_nodes)-1 else activation_last
            hl = KerasBase.activation(act)(hl)
            # dropout
            drp_rt = 0 if len(hidden_dropout_rates)==0 else hidden_dropout_rates[ih]
            hl = Dropout(drp_rt)(hl)
        
        #add
        hl = Add()([multiplied, hl])

        #merge layer
        if merge=='sum':
            self.__count_call_sum_model += 1
            crs_trm = KerasBase.sum_layer(name='sum' + str(self.__count_call_sum_model))(hl)
        elif merge=='mean':
            self.__count_call_mean_model += 1
            crs_trm = KerasBase.mean_layer(name='mean' + str(self.__count_call_mean_model))(hl)

        return crs_trm


    def compile(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss)
        return

    def fit(self, user_ids, item_ids, rating, batch_size, epochs, 
                  user_ids_val=None, item_ids_val=None, rating_val=None):
        
        # validation data
        val_data = None
        if (user_ids_val is not None) and (item_ids_val is not None) and (rating_val is not None):
            val_data = ([user_ids_val, item_ids_val], rating_val)

        # fit
        self.history = self.model.fit(x=[user_ids, item_ids], y=rating, 
                                      batch_size=batch_size, epochs=epochs, verbose=1, 
                                      validation_data=val_data)

        return

    def predict(self, user_ids, item_ids):
        return self.model.predict([user_ids, item_ids])[:,0]


import numpy as np

class MatrixFactorization:
    def __init__(self, latent_num):
        self.latent_num = latent_num
        
        # r = mu + u_bias + i_bias + dot(u_latent,i_latent)
        self.mu = None
        self.u_bias = None
        self.i_bias = None
        self.u_latent = None
        self.i_latent = None

        # id_index_dict
        self.user_id_index_dict = None
        self.item_id_index_dict = None

        return

    def fit(self, user_ids, item_ids, rating, batch_size, epochs, lerning_rate=0.1, l2=0):
        print('run MatrixFactorization fit')
        # num
        user_num = len(np.unique(user_ids))
        item_num = len(np.unique(item_ids))
        sample_num = len(user_ids)

        # id_index_dict
        self.user_id_index_dict = self.id_index_dict(np.unique(user_ids))
        self.item_id_index_dict = self.id_index_dict(np.unique(item_ids))

        # make index
        user_idxs = self.convert_ids_to_index(user_ids, self.user_id_index_dict)
        item_idxs = self.convert_ids_to_index(item_ids, self.item_id_index_dict)
        
        # mu
        self.mu = np.average(rating)

        # initialization
        self.u_bias = self.__initialization_bias(user_num)
        self.i_bias = self.__initialization_bias(item_num)
        self.u_latent = self.__initialization_latent(user_num, self.latent_num)
        self.i_latent = self.__initialization_latent(item_num, self.latent_num)

        # calc
        errors_in_epoch = self.__minibatch_sgd(user_idxs, item_idxs, rating, batch_size, epochs, lerning_rate, l2)

        return

    def __initialization_bias(self,  id_num):
        # itnialize -0.05 ~ 0.05
        b = (np.random.rand(id_num) - 0.5) * 0.1
        return b

    def __initialization_latent(self, id_num, latent_num):
        '''
        return latent (shape=[id_num, latent_num])
        '''
        # itnialize -0.05 ~ 0.05
        lt = (np.random.rand(id_num, latent_num) - 0.5) * 0.1
        return lt

    def __minibatch_sgd(self, user_idxs, item_idxs, rating, batch_size, epochs, lerning_rate=0.1, l2=0, verbose=True):
        #
        sample_num = len(user_idxs)
        steps_per_epoch = int(np.ceil(len(user_idxs) / batch_size))
        loss_in_epoch = []
        error_in_epoch = []
        
        ## for epoch
        for iep in range(epochs):
            rand_sample_idxs = np.random.permutation(np.arange(sample_num))
            ## for steps_per_epoch
            for istp in range(steps_per_epoch):
                # indexs in this mini batch
                batch_idxs = rand_sample_idxs[batch_size*istp : batch_size*(istp+1)]
                # update
                delta_u_bias, delta_i_bias, delta_u_latent, delta_i_latent = self.__delta_param(user_idxs[batch_idxs], item_idxs[batch_idxs], rating[batch_idxs])
                self.u_bias += lerning_rate * (delta_u_bias - l2 * self.u_bias)
                self.i_bias += lerning_rate * (delta_i_bias - l2 * self.i_bias)
                self.u_latent += lerning_rate * (delta_u_latent - l2 * self.u_latent)
                self.i_latent += lerning_rate * (delta_i_latent - l2 * self.i_latent)
            # recording error
            loss_in_epoch.append(self.__loss_function(user_idxs, item_idxs, rating, l2))
            error_in_epoch.append(self.__error_function(user_idxs, item_idxs, rating))
            # verbose
            print(' epoch {0}: error = {1:.4f}, loss = {2:.4f}'.format(iep+1, error_in_epoch[iep], loss_in_epoch[iep]))
        
        return error_in_epoch

    def __delta_param(self, user_idxs, item_idxs, rating):
        #
        delta_u_bias = np.zeros_like(self.u_bias)
        delta_i_bias = np.zeros_like(self.i_bias)
        delta_u_latent = np.zeros_like(self.u_latent)
        delta_i_latent = np.zeros_like(self.i_latent)

        #
        loss = rating - self.__calc_rating(user_idxs, item_idxs)
        #
        num_sample = len(user_idxs)

        #
        u_counter = np.zeros_like(self.u_bias)
        i_counter = np.zeros_like(self.i_bias)
        
        # calculate delta
        for ismp in range(num_sample):
            u_idx = user_idxs[ismp]
            i_idx = item_idxs[ismp]
            ls = loss[ismp]
            #
            delta_u_bias[u_idx] += ls
            delta_i_bias[i_idx] += ls
            delta_u_latent[u_idx] += ls * self.i_latent[i_idx]
            delta_i_latent[i_idx] += ls * self.u_latent[u_idx]
            #
            u_counter[u_idx] += 1
            i_counter[i_idx] += 1
        # average delta
        u_counter = np.maximum(u_counter, 1)
        i_counter = np.maximum(i_counter, 1)
        #u_counter = np.maximum(u_counter, num_sample)
        #i_counter = np.maximum(i_counter, num_sample)
        delta_u_bias /= u_counter
        delta_i_bias /= i_counter
        delta_u_latent /= u_counter[:,np.newaxis]
        delta_i_latent /= i_counter[:,np.newaxis]

        return delta_u_bias, delta_i_bias, delta_u_latent, delta_i_latent

    def __loss_function(self, user_idxs, item_idxs, rating, l2):
        e = np.sum(np.square(rating - self.__calc_rating(user_idxs, item_idxs)))
        l = l2 * (np.sum(np.square(self.u_bias[user_idxs])) + np.sum(np.square(self.i_bias[item_idxs]))
                 + np.sum(np.square(self.u_latent[user_idxs]))+ + np.sum(np.square(self.i_latent[item_idxs])))
        e_l = e + l

        return e_l
    def __error_function(self, user_idxs, item_idxs, rating):
        e = np.average(np.square(rating - self.__calc_rating(user_idxs, item_idxs)))

        return e


    def __calc_rating(self, user_idxs, item_idxs):
        mu = self.mu
        u_b = self.u_bias[user_idxs]
        i_b = self.i_bias[item_idxs]
        t1 = self.u_latent[user_idxs]
        t2 = self.i_latent[item_idxs]
        cross_term = np.sum(self.u_latent[user_idxs] * self.i_latent[item_idxs], axis=1)

        rt = mu + u_b + i_b + cross_term
        return rt

    def predict(self, user_ids, item_ids):
        # make index
        user_idxs = self.convert_ids_to_index(user_ids, self.user_id_index_dict)
        item_idxs = self.convert_ids_to_index(item_ids, self.item_id_index_dict)
        
        rt = self.__calc_rating(user_idxs, item_idxs)
        return rt

    @staticmethod
    def id_index_dict(unique_id_list):
        '''
        return id_index_dic = { id0 : 0, id01 : 1, ... }
        '''
        num = len(unique_id_list)
        id_idx_dict = dict(zip(unique_id_list, list(range(num))))
        return id_idx_dict

    @staticmethod
    def convert_id_to_index(id, id_idx_dict):
        idx = id_idx_dict.get(id)
        return idx

    @staticmethod
    def convert_ids_to_index(ids, id_idx_dict):
        idxs = []
        for id in ids:
            idx = id_idx_dict.get(id)
            idxs.append(idx)
        return np.array(idxs, dtype='int')








