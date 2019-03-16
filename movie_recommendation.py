import movieLens
import recommendation_base
import deepMatrixFactorization as dmf

import numpy as np
import pandas as pd
import os

class MovieRecommendation:

    @staticmethod
    def test_random_base(datafilename='ratings_test.csv'):
        target_id = 'movieId'

        rating_df_train, rating_df_test = MovieRecommendation.dataset_1(datafilename=datafilename)
        #
        rcmd = recommendation_base.BaseOnRandom()
        rcmd.fit(rating_df_train[target_id].unique(), rating_min=0.5, rating_max=5, seed=100)
        
        #
        pre_rating = rcmd.predict(rating_df_train[target_id].values)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(rating_df_test[target_id].values)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('BaseOnRandom', rmse_train, rmse_test)

        return

    @staticmethod
    def test_allmean_base(datafilename='ratings_test.csv'):
        target_id = 'movieId'

        rating_df_train, rating_df_test = MovieRecommendation.dataset_1(datafilename=datafilename)
        #
        rcmd = recommendation_base.BaseOnConstant()
        rcmd.fit(rating_df_train['rating'].mean())
        
        #
        pre_rating = rcmd.predict(rating_df_train[target_id].values)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(rating_df_test[target_id].values)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('BaseOnAllmean', rmse_train, rmse_test)

        return


    @staticmethod
    def test_mean_base_item(datafilename='ratings_test.csv'):
        target_id = 'movieId'

        rating_df_train, rating_df_test = MovieRecommendation.dataset_1(datafilename=datafilename)
        #
        rcmd = recommendation_base.BaseOnMean()
        rcmd.fit(rating_df_train[target_id].values, rating_df_train['rating'].values)
        
        #
        pre_rating = rcmd.predict(rating_df_train[target_id].values)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(rating_df_test[target_id].values)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('BaseOnMovieMean', rmse_train, rmse_test)

        return

    @staticmethod
    def test_mean_base_user(datafilename='ratings_test.csv'):
        target_id = 'userId'

        rating_df_train, rating_df_test = MovieRecommendation.dataset_1(datafilename=datafilename)
        #
        rcmd = recommendation_base.BaseOnMean()
        rcmd.fit(rating_df_train[target_id].values, rating_df_train['rating'].values)
        
        #
        pre_rating = rcmd.predict(rating_df_train[target_id].values)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(rating_df_test[target_id].values)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('BaseOnUserMean', rmse_train, rmse_test)

        return

    @staticmethod
    def test_nmf_1(n_components=2):
        rating_df_train, rating_df_test = MovieRecommendation.dataset_1()
        rating_mtrx_train = movieLens.RatingDataSet.rating_matrix(rating_df_train)
        rating_mtrx_test = movieLens.RatingDataSet.rating_matrix(rating_df_test)
        #
        rcmd = recommendation_base.NonNegaMF(n_components=n_components)
        rcmd.fit(rating_mtrx_train)
        #
        pre_rating = rcmd.predict(rating_df_train['userId'].values, rating_df_train['movieId'].values)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(rating_df_test['userId'].values, rating_df_test['movieId'].values)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('NonNegaMF', rmse_train, rmse_test)

        return

    @staticmethod
    def test_mf_1(datafilename='ratings_test.csv', epochs=10, cross_term_l2=0):
        # data
        rating_df_train, rating_df_test, userIndex_train, movieIndex_train, userIndex_test, movieIndex_test, _, _ = MovieRecommendation.dataset_1_for_mf(datafilename)
        
        # define
        unique_user_num = len(rating_df_train['userId'].unique())
        unique_item_num = len(rating_df_train['movieId'].unique())
        all_rating_mean = rating_df_train['rating'].mean()
        rating_scale = rating_df_train['rating'].std()
        rcmd = dmf.DeepMatrixFactorization(unique_user_num, unique_item_num, all_rating_mean, rating_scale=rating_scale)

        # make model
        rcmd.make_model_mf(user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=cross_term_l2)

        # compile
        rcmd.compile(optimizer='adam', loss='mean_squared_error')

        # model visualization
        output_dir = dmf.KerasBase.make_output_dir(base_dir_name=os.path.join('.','result'), dir_name='result', with_datetime=True)
        dmf.KerasBase.model_summary(rcmd.model, save_filename=os.path.join(output_dir,'summary.txt'), print_console=True)
        dmf.KerasBase.model_visualize(rcmd.model, save_filename=os.path.join(output_dir,'model_img.png'), show_shapes=True, show_layer_names=True)

        # fit
        rcmd.fit(userIndex_train, movieIndex_train, rating_df_train['rating'].values, batch_size=128, epochs=epochs, 
                 user_ids_val=userIndex_test, item_ids_val=movieIndex_test, rating_val=rating_df_test['rating'].values)

        #
        pre_rating = rcmd.predict(userIndex_train, movieIndex_train)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(userIndex_test, movieIndex_test)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('MF', rmse_train, rmse_test)

        #
        dmf.KerasBase.save_learning_history_loss(rcmd.history, val=True, filename=os.path.join(output_dir,'learning_hist.png'), show=False)


        return
    def test_mf_2(datafilename='ratings_test.csv', epochs=10, cross_term_l2=0, hidden_l2=[], hidden_dropout_rates=[]):
        # data
        rating_df_train, rating_df_test, userIndex_train, movieIndex_train, userIndex_test, movieIndex_test, _, _ = MovieRecommendation.dataset_1_for_mf(datafilename)
        
        # define
        unique_user_num = len(rating_df_train['userId'].unique())
        unique_item_num = len(rating_df_train['movieId'].unique())
        all_rating_mean = rating_df_train['rating'].mean()
        rating_scale = rating_df_train['rating'].std()
        rcmd = dmf.DeepMatrixFactorization(unique_user_num, unique_item_num, all_rating_mean, rating_scale=rating_scale)

        # make model
        rcmd.make_model_dmf_deepLatent(user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=cross_term_l2, hidden_nodes_latent=[10], hidden_l2=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)
        #rcmd.make_model_mf(user_bias=True, item_bias=True, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=True, latent_num=10, cross_term_l2=0)

        # compile
        rcmd.compile(optimizer='adam', loss='mean_squared_error')

        # model visualization
        output_dir = dmf.KerasBase.make_output_dir(base_dir_name=os.path.join('.','result'), dir_name='result', with_datetime=True)
        dmf.KerasBase.model_summary(rcmd.model, save_filename=os.path.join(output_dir,'summary.txt'), print_console=True)
        dmf.KerasBase.model_visualize(rcmd.model, save_filename=os.path.join(output_dir,'model_img.png'), show_shapes=True, show_layer_names=True)

        # fit
        rcmd.fit(userIndex_train, movieIndex_train, rating_df_train['rating'].values, batch_size=128, epochs=epochs, 
                 user_ids_val=userIndex_test, item_ids_val=movieIndex_test, rating_val=rating_df_test['rating'].values)

        #
        pre_rating = rcmd.predict(userIndex_train, movieIndex_train)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(userIndex_test, movieIndex_test)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('DMF_lt', rmse_train, rmse_test)

        #
        dmf.KerasBase.save_learning_history_loss(rcmd.history, val=True, filename=os.path.join(output_dir,'learning_hist.png'), show=False)


        return
    def test_mf_3(datafilename='ratings_test.csv', epochs=10, cross_term_l2=0, hidden_l2=[], hidden_dropout_rates=[]):
        # data
        rating_df_train, rating_df_test, userIndex_train, movieIndex_train, userIndex_test, movieIndex_test, _, _ = MovieRecommendation.dataset_1_for_mf(datafilename)
        
        # define
        unique_user_num = len(rating_df_train['userId'].unique())
        unique_item_num = len(rating_df_train['movieId'].unique())
        all_rating_mean = rating_df_train['rating'].mean()
        rating_scale = rating_df_train['rating'].std()
        rcmd = dmf.DeepMatrixFactorization(unique_user_num, unique_item_num, all_rating_mean, rating_scale=rating_scale)

        # make model
        rcmd.make_model_dmf_deepCrossterm(user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=cross_term_l2, hidden_nodes_crossterm=[10], hidden_l2=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)
        #rcmd.make_model_mf(user_bias=True, item_bias=True, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=True, latent_num=10, cross_term_l2=0)

        # compile
        rcmd.compile(optimizer='adam', loss='mean_squared_error')

        # model visualization
        output_dir = dmf.KerasBase.make_output_dir(base_dir_name=os.path.join('.','result'), dir_name='result', with_datetime=True)
        dmf.KerasBase.model_summary(rcmd.model, save_filename=os.path.join(output_dir,'summary.txt'), print_console=True)
        dmf.KerasBase.model_visualize(rcmd.model, save_filename=os.path.join(output_dir,'model_img.png'), show_shapes=True, show_layer_names=True)

        # fit
        rcmd.fit(userIndex_train, movieIndex_train, rating_df_train['rating'].values, batch_size=128, epochs=epochs, 
                 user_ids_val=userIndex_test, item_ids_val=movieIndex_test, rating_val=rating_df_test['rating'].values)

        #
        pre_rating = rcmd.predict(userIndex_train, movieIndex_train)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(userIndex_test, movieIndex_test)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('DMF_ct', rmse_train, rmse_test)

        #
        dmf.KerasBase.save_learning_history_loss(rcmd.history, val=True, filename=os.path.join(output_dir,'learning_hist.png'), show=False)


        return
    def test_mf_4(datafilename='ratings_test.csv', epochs=10, cross_term_l2=0, hidden_l2=[], hidden_dropout_rates=[]):
        # data
        rating_df_train, rating_df_test, userIndex_train, movieIndex_train, userIndex_test, movieIndex_test, _, _ = MovieRecommendation.dataset_1_for_mf(datafilename)
        
        # define
        unique_user_num = len(rating_df_train['userId'].unique())
        unique_item_num = len(rating_df_train['movieId'].unique())
        all_rating_mean = rating_df_train['rating'].mean()
        rating_scale = rating_df_train['rating'].std()
        rcmd = dmf.DeepMatrixFactorization(unique_user_num, unique_item_num, all_rating_mean, rating_scale=rating_scale)

        # make model
        rcmd.make_model_dmf_deepLatent_deepCrossterm(user_bias=True, item_bias=True, cross_term=True, latent_num=10, 
                                                     cross_term_l2=cross_term_l2, hidden_nodes_latent=[10], hidden_nodes_crossterm=[10], hidden_l2=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)
        #rcmd.make_model_mf(user_bias=True, item_bias=True, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=True, latent_num=10, cross_term_l2=0)

        # compile
        rcmd.compile(optimizer='adam', loss='mean_squared_error')

        # model visualization
        output_dir = dmf.KerasBase.make_output_dir(base_dir_name=os.path.join('.','result'), dir_name='result', with_datetime=True)
        dmf.KerasBase.model_summary(rcmd.model, save_filename=os.path.join(output_dir,'summary.txt'), print_console=True)
        dmf.KerasBase.model_visualize(rcmd.model, save_filename=os.path.join(output_dir,'model_img.png'), show_shapes=True, show_layer_names=True)

        # fit
        rcmd.fit(userIndex_train, movieIndex_train, rating_df_train['rating'].values, batch_size=128, epochs=epochs, 
                 user_ids_val=userIndex_test, item_ids_val=movieIndex_test, rating_val=rating_df_test['rating'].values)

        #
        pre_rating = rcmd.predict(userIndex_train, movieIndex_train)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(userIndex_test, movieIndex_test)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('DMF_lt_cs', rmse_train, rmse_test)

        #
        dmf.KerasBase.save_learning_history_loss(rcmd.history, val=True, filename=os.path.join(output_dir,'learning_hist.png'), show=False)


        return
    def test_mf_5(datafilename='ratings_test.csv', epochs=10, cross_term_l2=0, hidden_l2=[], hidden_dropout_rates=[]):
        # data
        rating_df_train, rating_df_test, userIndex_train, movieIndex_train, userIndex_test, movieIndex_test, _, _ = MovieRecommendation.dataset_1_for_mf(datafilename)
        
        # define
        unique_user_num = len(rating_df_train['userId'].unique())
        unique_item_num = len(rating_df_train['movieId'].unique())
        all_rating_mean = rating_df_train['rating'].mean()
        rating_scale = rating_df_train['rating'].std()
        rcmd = dmf.DeepMatrixFactorization(unique_user_num, unique_item_num, all_rating_mean, rating_scale=rating_scale)

        # make model
        rcmd.make_model_dmf_residualDeepCrossterm(user_bias=True, item_bias=True, cross_term=True, latent_num=10, cross_term_l2=cross_term_l2, hidden_nodes_crossterm=[10], hidden_l2=hidden_l2, hidden_dropout_rates=hidden_dropout_rates)
        #rcmd.make_model_mf(user_bias=True, item_bias=True, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=False, latent_num=10, cross_term_l2=0)
        #rcmd.make_model_mf(user_bias=True, item_bias=False, cross_term=True, latent_num=10, cross_term_l2=0)

        # compile
        rcmd.compile(optimizer='adam', loss='mean_squared_error')

        # model visualization
        output_dir = dmf.KerasBase.make_output_dir(base_dir_name=os.path.join('.','result'), dir_name='result', with_datetime=True)
        dmf.KerasBase.model_summary(rcmd.model, save_filename=os.path.join(output_dir,'summary.txt'), print_console=True)
        dmf.KerasBase.model_visualize(rcmd.model, save_filename=os.path.join(output_dir,'model_img.png'), show_shapes=True, show_layer_names=True)

        # fit
        rcmd.fit(userIndex_train, movieIndex_train, rating_df_train['rating'].values, batch_size=128, epochs=epochs, 
                 user_ids_val=userIndex_test, item_ids_val=movieIndex_test, rating_val=rating_df_test['rating'].values)

        #
        pre_rating = rcmd.predict(userIndex_train, movieIndex_train)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(userIndex_test, movieIndex_test)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('DMF_res_ct', rmse_train, rmse_test)

        #
        dmf.KerasBase.save_learning_history_loss(rcmd.history, val=True, filename=os.path.join(output_dir,'learning_hist.png'), show=False)


        return
    
    
    def test_mf2_naive_implement(datafilename='ratings_test.csv'):
        # data
        rating_df_train, rating_df_test, userIndex_train, movieIndex_train, userIndex_test, movieIndex_test, _, _ = MovieRecommendation.dataset_1_for_mf(datafilename)
        
        # define
        rcmd = dmf.MatrixFactorization(latent_num=10)

        # fit
        rcmd.fit(userIndex_train, movieIndex_train, rating_df_train['rating'].values, 
                 batch_size=128, epochs=10, lerning_rate=0.1, l2=0)

        #
        pre_rating = rcmd.predict(userIndex_train, movieIndex_train)
        rmse_train = MovieRecommendation.rmse(pre_rating, rating_df_train['rating'].values)
        #
        pre_rating = rcmd.predict(userIndex_test, movieIndex_test)
        rmse_test = MovieRecommendation.rmse(pre_rating, rating_df_test['rating'].values)
        #
        MovieRecommendation.print_summary('naive MF', rmse_train, rmse_test)

        return

    @staticmethod
    def rmse(ar1, ar2):
        return np.sqrt(np.average(np.square(ar1-ar2)))

    @staticmethod
    def print_summary(title, rmse_train, rmse_test):
        print('<summary ' + title + '>')
        print(' rmse_train, rmse_test = {0:.5f}, {1:.5f}'.format(rmse_train, rmse_test))
        return

    @staticmethod
    def dataset_1(datafilename='ratings_test.csv', rating_num_threshold=10, train_rate=0.8, analysis=False):
        '''
        return pandas DataFrame that have columns = [userId, movieId, rating, timestamp]
        '''
        ############
        # constant
        ############
        RATING_NUM_THRESHOLD = rating_num_threshold
        FILE_PATH = os.path.join('.', 'movieLens_data', datafilename)
        
        ####################
        # movieLens data set
        ####################
        ml_data = movieLens.RatingDataSet()
        # read file
        rating_df = ml_data.read_rating(FILE_PATH)
        # delete small number of rating
        rating_df = ml_data.delete_small_number_rating(rating_df, RATING_NUM_THRESHOLD, target='userId')
        rating_df = ml_data.delete_small_number_rating(rating_df, RATING_NUM_THRESHOLD, target='movieId')
        rating_df = ml_data.delete_small_number_rating(rating_df, RATING_NUM_THRESHOLD, target='userId')
        # analysis
        if analysis:
            ml_data.rating_analysis(rating_df)
    
        ########################
        # train and test data
        ########################
        # split
        rating_df_train, rating_df_test = ml_data.train_test_split_rating_df(rating_df, train_rate=train_rate, target='userId', random_state=200)
        # rating_df_train, rating_df_test = ml_data.train_test_split_rating_df(rating_df, train_rate=0.8, target='movieId')
        # delete small number of rating
        rating_df_train = ml_data.delete_small_number_rating(rating_df_train, RATING_NUM_THRESHOLD, target='userId')
        rating_df_train = ml_data.delete_small_number_rating(rating_df_train, RATING_NUM_THRESHOLD, target='movieId')
        rating_df_train = ml_data.delete_small_number_rating(rating_df_train, RATING_NUM_THRESHOLD, target='userId')
        #
        rating_df_test = ml_data.delete_small_number_rating(rating_df_test, RATING_NUM_THRESHOLD, target='userId')
        rating_df_test = ml_data.delete_small_number_rating(rating_df_test, RATING_NUM_THRESHOLD, target='movieId')
        rating_df_test = ml_data.delete_small_number_rating(rating_df_test, RATING_NUM_THRESHOLD, target='userId')
        #
        rating_df_test = ml_data.delete_include_only_test_rating_df(rating_df_train, rating_df_test, target='userId')
        rating_df_test = ml_data.delete_include_only_test_rating_df(rating_df_train, rating_df_test, target='movieId')
        
        # analysis
        if analysis:
            ml_data.rating_analysis(rating_df_train)
            ml_data.rating_analysis(rating_df_test)

        ################
        # rating matrix
        ################
        #rating_mtrx_train = ml_data.rating_matrix(rating_df_train)
        #rating_mtrx_test = ml_data.rating_matrix(rating_df_test)

        #return rating_df_train, rating_df_test, rating_mtrx_train, rating_mtrx_test
        return rating_df_train, rating_df_test

    @staticmethod
    def dataset_1_for_mf(datafilename='ratings_test.csv', rating_num_threshold=10, analysis=False):
        # rating df
        rating_df_train, rating_df_test = MovieRecommendation.dataset_1(datafilename=datafilename, rating_num_threshold=rating_num_threshold, analysis=analysis)

        # id_index_dic = { id0 : 0, id01 : 1, ... }
        userId_index_dict = movieLens.RatingDataSet.id_index_dict(rating_df_train['userId'].unique())
        movieId_index_dict = movieLens.RatingDataSet.id_index_dict(rating_df_train['movieId'].unique())

        unique_train = pd.Series(rating_df_train['movieId'].unique())
        unique_test = pd.Series(rating_df_test['movieId'].unique())

        # index
        userIndex_train = movieLens.RatingDataSet.convert_id_to_index(rating_df_train['userId'].values, userId_index_dict)
        movieIndex_train = movieLens.RatingDataSet.convert_id_to_index(rating_df_train['movieId'].values, movieId_index_dict)
        userIndex_test = movieLens.RatingDataSet.convert_id_to_index(rating_df_test['userId'].values, userId_index_dict)
        movieIndex_test = movieLens.RatingDataSet.convert_id_to_index(rating_df_test['movieId'].values, movieId_index_dict)

        return rating_df_train, rating_df_test, userIndex_train, movieIndex_train, userIndex_test, movieIndex_test, userId_index_dict, movieId_index_dict


#MovieRecommendation.test_mf_1(datafilename='ratings.csv', cross_term_l2=0.00001, epochs=10)
#MovieRecommendation.test_mf_2(datafilename='ratings.csv', cross_term_l2=0.00001, epochs=10, hidden_l2=[0.000001])
#MovieRecommendation.test_mf_3(datafilename='ratings.csv', cross_term_l2=0.00001, epochs=10, hidden_l2=[0.0000001])
#MovieRecommendation.test_mf_4(datafilename='ratings.csv', cross_term_l2=0.000001, epochs=10, hidden_l2=[0.00000001])
#MovieRecommendation.test_mf_5(datafilename='ratings.csv', cross_term_l2=0.0001, epochs=10, hidden_l2=[0.00001], hidden_dropout_rates=[0])

k1 = 0.1
k2 = (85570+69440)/(2319780+218510)
MovieRecommendation.test_mf_1(datafilename='ratings.csv', cross_term_l2=k1*0.00001, epochs=10)
MovieRecommendation.test_mf_2(datafilename='ratings.csv', cross_term_l2=k1*0.00001, epochs=10, hidden_l2=[k2*0.000001])
MovieRecommendation.test_mf_3(datafilename='ratings.csv', cross_term_l2=k1*0.00001, epochs=10, hidden_l2=[k2*0.0000001])
MovieRecommendation.test_mf_4(datafilename='ratings.csv', cross_term_l2=k1*0.000001, epochs=10, hidden_l2=[k2*0.00000001])
MovieRecommendation.test_mf_5(datafilename='ratings.csv', cross_term_l2=k1*0.0001, epochs=10, hidden_l2=[k2*0.00001], hidden_dropout_rates=[0])


#MovieRecommendation.test_mf2_naive_implement()
MovieRecommendation.test_random_base(datafilename='ratings.csv')
MovieRecommendation.test_allmean_base(datafilename='ratings.csv')
MovieRecommendation.test_mean_base_item(datafilename='ratings.csv')
MovieRecommendation.test_mean_base_user(datafilename='ratings.csv')
