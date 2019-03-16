import pandas as pd
import numpy as np

from sklearn.decomposition import NMF

class BaseOnRandom:
    def __init__(self):
        self.rating_df = None
        self.id_index_dic = None
        self.index_rating_dic = None
        return

    def fit(self, unique_id, rating_min, rating_max, seed=None):
        #
        unique_id_num = len(unique_id)
        #
        np.random.seed(seed)
        random_rt = np.random.rand(unique_id_num) * (rating_max - rating_min) + rating_min
        
        #
        self.rating_df = pd.concat([pd.DataFrame(unique_id), pd.DataFrame(random_rt)], axis=1)
        self.rating_df.columns = ['id', 'rating']

        #
        tempdf = self.rating_df['id'].reset_index() # index, id
        tempdf.index = tempdf['id'] # id, index, id
        self.id_index_dic = tempdf['index'].to_dict() # key=id, value=index

        #
        self.index_rating_dic = self.rating_df['rating'].to_dict() # key=index, value=rating

        return

    def predict(self, id):
        def search(_id):
            _idx = self.id_index_dic.get(_id)
            _rt = self.index_rating_dic.get(_idx) if _idx is not None else 0
            _rt = _rt if _rt is not None else 0
            return _rt
        vecf_search = np.vectorize(search)
        #
        pre_rating = vecf_search(id)
        return pre_rating

class BaseOnConstant:
    def __init__(self):
        self.constant = None
        return

    def fit(self, constant):
        self.constant = constant
        return

    def predict(self, id):
        pre_rating = np.ones_like(id) * self.constant
        return pre_rating

class BaseOnMean:
    def __init__(self):
        self.rating_df = None
        self.id_index_dic = None
        self.index_rating_dic = None
        return

    def fit(self, id, rating):
        #
        df = pd.concat([pd.DataFrame(id), pd.DataFrame(rating)], axis=1)
        df.columns = ['id', 'rating']
        #
        grouped = df.groupby('id').mean()
        #
        self.rating_df = grouped.reset_index() # id, rating
        
        #
        tempdf = self.rating_df['id'].reset_index() # index, id
        tempdf.index = tempdf['id'] # id, index, id
        self.id_index_dic = tempdf['index'].to_dict() # key=id, value=index

        #
        self.index_rating_dic = self.rating_df['rating'].to_dict() # key=index, value=rating

        return

    def predict(self, id):
        def search(_id):
            _idx = self.id_index_dic.get(_id)
            _rt = self.index_rating_dic.get(_idx) if _idx is not None else 0
            _rt = _rt if _rt is not None else 0
            return _rt
        vecf_search = np.vectorize(search)
        #
        pre_rating = vecf_search(id)
        return pre_rating

class NonNegaMF:
    '''
    Non-negative Matrix Factorization: NMF
    '''
    def __init__(self, n_components):
        self.n_components = n_components

        #
        self.nmf_rt_mtrx = None
        self.dfindex_index_dic = None
        self.dfcolumn_index_dic = None

        return

    def fit(self, rating_mtrx_df):
        #
        tempdf = pd.DataFrame(rating_mtrx_df.index).reset_index() #index, 0
        self.dfindex_index_dic = (tempdf.set_index(rating_mtrx_df.index.name)['index']).to_dict()
        #
        tempdf = pd.DataFrame(rating_mtrx_df.columns).reset_index() #index, 0
        self.dfcolumn_index_dic = (tempdf.set_index(rating_mtrx_df.columns.name)['index']).to_dict()

        #
        nmf_model = NMF(n_components=self.n_components, verbose=False, max_iter=200)
        w = nmf_model.fit_transform(rating_mtrx_df.values)
        h = nmf_model.components_
        #
        self.nmf_rt_mtrx = np.dot(w, h)
        
        return

    def predict(self, index_id, column_id):
        
        def search(idx_id_, clmn_id_):
            #
            idx_ = self.dfindex_index_dic.get(idx_id_)
            clmn_ = self.dfcolumn_index_dic.get(clmn_id_)
            #
            rt = 0
            if (idx_ is not None) and (clmn_ is not None):
                rt = self.nmf_rt_mtrx[idx_, clmn_]
            else:
                rt = 0
            return rt
        vecf_search = np.vectorize(search)
        #
        pre_rating = vecf_search(index_id, column_id)
        return pre_rating






