import pandas as pd
import numpy as np
import sklearn.model_selection as skms

class RatingDataSet:
    def __init__(self):
        '''
        userId, movieId, rating, timestamp
        1, 307, 3.5, 1256677221
        1, 481, 3.5, 1256677456
        1, 1091, 1.5, 1256677471
        '''

        return

    @staticmethod
    def read_rating(filepth):
        '''
        read rating csv file
        format is as follows.
        userId, movieId, rating, timestamp
        1, 307, 3.5, 1256677221
        1, 481, 3.5, 1256677456
        1, 1091, 1.5, 1256677471
        '''
        #print('\nread_rating start')
        rating_df = pd.read_csv(filepth)
        rating_df = rating_df.sort_values(['userId', 'timestamp'])
        rating_df = rating_df.reset_index(drop=True)
        #print('read_rating end')
        return rating_df

    @staticmethod
    def rating_analysis(rating_df, rating_num_thresholds=list(range(20))):
        
        print('<rating analysis>')
        ############
        # about all
        ############
        #nan
        print('  num of nan : {0}'.format(rating_df.isnull().values.sum()))
        print()

        #############
        # about user
        #############
        print(' <user>')
        grouped = rating_df.groupby('userId')
        #rating num
        rating_num = grouped.size()
        RatingDataSet.__rating_analysis_num(rating_num, rating_num_thresholds)
        print()

        #############
        # about movie
        #############
        print(' <movie>')
        grouped = rating_df.groupby('movieId')
        #rating num
        rating_num = grouped.size()
        RatingDataSet.__rating_analysis_num(rating_num, rating_num_thresholds)
        print()

        return

        return
    
    @staticmethod
    def __rating_analysis_num(rating_num, rating_num_thresholds=list(range(20))):
        # unique_num
        unique_num = len(rating_num)
        print('  unique_num : {0}'.format(unique_num))

        # total_rating_num
        total_rating_num = rating_num.sum()
        print('  total_rating_num : {0}'.format(total_rating_num))

        # rating_num_min, max
        rating_num_min = rating_num.min()
        rating_num_max = rating_num.max()
        print('  rating_num_min, max : {0}, {1}'.format(rating_num_min, rating_num_max))

        # 
        print('  rating_num<, num, num/total_num')
        for threshold in rating_num_thresholds:
            un = (rating_num < threshold).sum()
            print('  {0}, {1}, {2:.2f}'.format(threshold, un, un/len(rating_num)))

        return

    @staticmethod
    def train_test_split_rating_df(rating_df, train_rate=0.8, target='userId', random_state=None):
        #print('\ntrain_test_split_rating_df start')
        #
        rating_df_train, rating_df_test = skms.train_test_split(rating_df, test_size=1.0-train_rate, stratify=rating_df[target], random_state=random_state )
        
        #
        rating_df_test = RatingDataSet.delete_include_only_test_rating_df(rating_df_train, rating_df_test, target='userId')
        rating_df_test = RatingDataSet.delete_include_only_test_rating_df(rating_df_train, rating_df_test, target='movieId')

        #sort
        rating_df_train = rating_df_train.sort_values(['userId', 'timestamp'])
        rating_df_train = rating_df_train.reset_index(drop=True)
        rating_df_test = rating_df_test.sort_values(['userId', 'timestamp'])
        rating_df_test = rating_df_test.reset_index(drop=True)
        
        #print('train_test_split_rating_df end')

        return rating_df_train, rating_df_test

    @staticmethod
    def delete_include_only_test_rating_df(rating_df_train, rating_df_test, target='userId'):
        unique_train = pd.Series(rating_df_train[target].unique())
        unique_test = pd.Series(rating_df_test[target].unique())
        #
        inc_only_test = unique_test[~unique_test.isin(unique_train)]
        #
        deleted_df = rating_df_test.copy()
        if len(inc_only_test) > 0:
            deleted_df = rating_df_test.drop(rating_df_test.index[rating_df_test[target].isin(inc_only_test)])
        
        return deleted_df

    @staticmethod
    def delete_small_number_rating(rating_df, threshold_num, target='userId'):
        '''
        item = 'userId' or 'movieId'
        extract userId (rating num >= threshold_num)
        '''
        #print('\ndelete_small_number_rating start')

        rating_df_ = rating_df.copy()
        #
        grouped = rating_df_.groupby(target)
        rating_num = grouped.size()
        #
        satisfy_index = rating_num.index[rating_num >= threshold_num]
        #
        rating_df_ = rating_df_[rating_df_[target].isin(satisfy_index)]
        rating_df_ = rating_df_.sort_values(['userId', 'timestamp'])
        rating_df_ = rating_df_.reset_index(drop=True)
        #print('delete_small_number_rating end')

        return rating_df_

    @staticmethod
    def id_index_dict(unique_id_list):
        '''
        return id_index_dic = { id0 : 0, id01 : 1, ... }
        '''
        num = len(unique_id_list)
        id_idx_dict = dict(zip(unique_id_list, list(range(num))))
        return id_idx_dict

    @staticmethod
    def convert_id_to_index(ids, id_idx_dict):
        idxs = []
        for id in ids:
            idx = id_idx_dict.get(id)
            idxs.append(idx)

        return np.array(idxs, dtype='int')

    #rating matrix
    @staticmethod
    def rating_matrix(rating_df):
        key1 = 'userId'
        key2 = 'movieId'
        key3 = 'rating'

        rating_mtrx_df = rating_df.pivot(index=key1, columns=key2, values=key3)
        rating_mtrx_df = rating_mtrx_df.fillna(0)

        return rating_mtrx_df

    @staticmethod
    def rating_average(rating_df, target='userId'):
        rating_ave = rating_df.groupby(target).mean()['rating'].values
        #print(rating_ave)
        return











