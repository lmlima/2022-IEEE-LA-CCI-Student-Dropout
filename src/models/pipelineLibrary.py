#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
import numpy as np
import pandas as pd


# In[2]:


map_forma_evasao = { 
    'Desistência':1,
    'Desligamento por Abandono':1,
    'Desligamento: Resolução 68/2017-CEPE':1,
    'Reopção de Curso':1,
    'Reopção de curso':1,
    'Transferência Interna':1,
    'Formado':0
}
map_semestre_disciplina = {
    '1º Semestre':'1º Semestre',
    '2º Semestre':'2º Semestre',
    '1. Trimestre':'1º Semestre', 
    '2. Trimestre':'1º Semestre',
    '3. Trimestre':'2º Semestre',
    'Curso de Verão':'1º Semestre'
}
map_forma_ingresso = {
    'Novo Curso':'OUTRAS',
    'Permanência':'OUTRAS',
    'Reopção de Curso':'OUTRAS', 
    'Transferência Facultativa':'OUTRAS',
    'SISU':'VESTIBULAR_OU_SISU',
    'Vestibular':'VESTIBULAR_OU_SISU'
}
map_nacionalidade = {
    'info_NACIONALIADE_0':'info_NACIONALIADE_0'
    # OUTRAS
}
map_naturalidade = {
    'info_NATURALIDADE_4':'NATURALIDADE_4',
    'info_NATURALIDADE_5':'NATURALIDADE_5'
    # OUTRAS
}
map_uf_naturalidade = {
    'info_UF_NATURALIDADE_0':'UF_NATURALIDADE_0',
    'info_UF_NATURALIDADE_1':'UF_NATURALIDADE_1'
    # OUTRAS
}
map_situacao_disciplina = {
    'Reprovado por Freqüência':'NÃO APROVADO POR FREQUENCIA',
    'Amparo Legal':'NÃO APROVADO',
    'Reprovado por Nota':'NÃO APROVADO',
    'Trancamento de Curso':'NÃO APROVADO',
    'Dispensa sem nota':'NÃO APROVADO',
    'Matrícula':'NÃO APROVADO',
    'Aprovado':'APROVADO',
    'Aprovado sem Nota':'APROVADO',
    'Aproveitamento de Estudos': 'APROVEITAMENTO',
    'Realizado':'APROVEITAMENTO',
    'Mobilidade Acadêmica':'APROVEITAMENTO',
    'Situação da atividade no currículo':'APROVEITAMENTO',
    # OUTRAS
}


# In[3]:


# ----- Transformer  -----
class MapTransformer():
    def __init__(self, feature_name, new_feature_name, dict_, fillna=False):
        self.feature_name = feature_name
        self.new_feature_name = new_feature_name
        self.dict_ = dict_
        self.fillna = fillna
    
    def transform(self, dataframe):
        df = dataframe.copy()
        if self.fillna:
            df[self.new_feature_name] = df[self.feature_name].map(self.dict_).fillna('OUTRAS')
        else:
            df[self.new_feature_name] = df[self.feature_name].map(self.dict_)
        return df
    
class RemoverPorMap():
    def __init__(self, feature_name, dict_):
        self.feature_name = feature_name
        self.dict_ = dict_
        
    def transform(self, df):
        df_ = df.copy()
        df_['FEATURE_VALIDA'] = df_[self.feature_name].map(self.dict_)
        return df[~df['ID_CURSO_ALUNO'].isin(df_[df_['FEATURE_VALIDA'].isnull()]['ID_CURSO_ALUNO'].unique())].copy()
    
class Remover():
    def __init__(self, feature_name, list_):
        self.feature_name = feature_name
        self.list_ = list_
        
    def transform(self, df):
        return df[~df[self.feature_name].isin(self.list_)].copy()
        
    
class FeatureSelection():
    def __init__(self, lt, inverse_selection=False):
        self.lt = lt
        self.inverse_selection = inverse_selection
        
    def transform(self, dataframe):
        if self.inverse_selection:
            return dataframe[dataframe.columns.difference(self.lt)].copy()
        else:
            return dataframe[self.lt].copy()
    
# ----- Extra -----     
class FeatureDuplicate():
    def __init__(self, feature_name, new_feature_name):
        self.feature_name = feature_name
        self.new_feature_name = new_feature_name
        
    def transform(self, df):
        df[self.new_feature_name] = df[self.feature_name]
        return df.copy()

# ----- Estimator -----
class Winsorization():
    def __init__(self, feature_name, new_feature_name):
        self.feature_name = feature_name
        self.new_feature_name = new_feature_name
        
    def fit_transform(self, dataframe):
        df = dataframe.copy()
        self.max_percentile = df[self.feature_name].quantile(0.95)
        self.min_percentile = df[self.feature_name].quantile(0.05)
        df[self.new_feature_name] = df[self.feature_name].apply(lambda x : self.max_percentile if x>self.max_percentile else x)                                                .apply(lambda x : self.min_percentile if x<self.min_percentile else x)
        return df
        
    def transform(self, dataframe):
        df = dataframe.copy()
        df[self.new_feature_name] = df[self.feature_name].apply(lambda x : self.max_percentile if x>self.max_percentile else x)                                                .apply(lambda x : self.min_percentile if x<self.min_percentile else x)
        return df
    
class FillMean():
    def __init__(self, feature):
        self.feature = feature

    def fit_transform(self, dataframe):
        df = dataframe.copy()
        self.mean = df[self.feature].mean()
        df[self.feature].fillna(self.mean, inplace=True)
        return df
        
    def transform(self, dataframe):
        df = dataframe.copy()
        df[self.feature].fillna(self.mean, inplace=True)
        return df
    
class StandardScaler():
    def __init__(self, features, inverse_selection=False):
        self.features = features
        self.inverse_selection = inverse_selection
    
    def fit_transform(self, dataframe):
        if self.inverse_selection:
            columns = dataframe[dataframe.columns.difference(self.features)].columns
            self.mean = dataframe[columns].mean()
            self.std = dataframe[columns].std()

            df = dataframe.copy()
            df.loc[:, columns] = (df[columns] - self.mean)/self.std
        else:
            self.mean = dataframe[self.features].mean()
            self.std = dataframe[self.features].std()

            df = dataframe.copy()
            df.loc[:, self.features] = (df[self.features] - self.mean)/self.std
        return df
    
    def transform(self, dataframe):
        if self.inverse_selection:
            columns = dataframe[dataframe.columns.difference(self.features)].columns
            df = dataframe.copy()
            df.loc[:, columns] = (df[columns] - self.mean)/self.std
        else:
            df = dataframe.copy()
            df.loc[:, self.features] = (df[self.features] - self.mean)/self.std
        return df
        
class LabelEncoder():
    def __init__(self, feature, dict_):
        self.feature = feature
        self.dict_ = dict_
        
    def transform(self, dataframe):
        df = dataframe.copy()
        df.loc[:,self.feature] = df[self.feature].map(self.dict_).fillna(-1.0)
        return df
    
class OneHotEncoder():
    def __init__(self, features):
        self.features = features
    
    def fit_transform(self, dataframe):
        self.enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        df_ = pd.DataFrame(self.enc.fit_transform(dataframe[self.features]), 
                           columns=self.enc.get_feature_names(self.features), dtype=np.int)
        return pd.concat([dataframe.reset_index(drop=True), df_], axis=1).copy()
    
    def transform(self, dataframe):
        df_ = pd.DataFrame(self.enc.transform(dataframe[self.features]), 
                           columns=self.enc.get_feature_names(self.features), dtype=np.int)
        return pd.concat([dataframe.reset_index(drop=True), df_], axis=1).copy() 
        
# ----- Pipeline -----      
class Pipeline():
    def __init__(self, args, filter_function=None):
        self.pipeline = args
        self.filter = filter_function
    
    def fit_transform(self, df):
        for process in self.pipeline:
            if hasattr(process, 'fit_transform'):
                df = process.fit_transform(df)
            else:
                df = process.transform(df)
        if (self.filter):
            return self.filter.transform(df)
        return df
         
    def transform(self, df):
        for process in self.pipeline:
            df = process.transform(df)
        if (self.filter):
            return self.filter.transform(df)
        return df
    
###################################################################################

class PipelineLSTM():
    def __init__(self, args, keys, reshape=None):
        self.pipeline = args
        self.keys = keys
        self.reshape = reshape
    
    def fit_transform(self, df):
        dataframe_main = df[self.keys].drop_duplicates().copy()
        for process in self.pipeline:
            if hasattr(process, 'fit_transform'):
                dataframe_main = process.fit_transform(dataframe_main)
            else:
                dataframe_main = process.transform(self.keys, dataframe_main, df)
        if self.reshape:
            return self.reshape.transform(dataframe_main)
        return dataframe_main
    
    def transform(self, df):
        dataframe_main = df[self.keys].drop_duplicates().copy()
        for process in self.pipeline:
            if hasattr(process, 'fit_transform'):
                dataframe_main = process.transform(dataframe_main)
            else:
                dataframe_main = process.transform(self.keys, dataframe_main, df)
        if self.reshape:
            return self.reshape.transform(dataframe_main)
        return dataframe_main
        
class SomaGroupBy():
    def __init__(self, columns_exception):
        self.columns_exception = columns_exception
        
    def transform(self, keys, dataframe_main, dataframe_static):
        df_ = dataframe_static.groupby(keys).sum().reset_index()
        df_ = df_[df_.columns.difference(self.columns_exception)]
        df_.columns = ['SOMA_'+column_name if column_name not in keys else column_name for column_name in df_.columns]
        return dataframe_main.merge(df_, how='left', on=keys).copy()
    
class DesvioGroupBy():
    def __init__(self, columns_exception):
        self.columns_exception = columns_exception
        
    def transform(self, keys, dataframe_main, dataframe_static):
        df_ = dataframe_static.groupby(keys).std().reset_index().fillna(0)
        df_ = df_[df_.columns.difference(self.columns_exception)]
        df_.columns = ['DESVIO_'+column_name if column_name not in keys else column_name for column_name in df_.columns]
        return dataframe_main.merge(df_, how='left', on=keys).copy()
    
class MediaGroupBy():
    def __init__(self, columns_exception):
        self.columns_exception = columns_exception
        
    def transform(self, keys, dataframe_main, dataframe_static):
        df_ = dataframe_static.groupby(keys).mean().reset_index()
        df_ = df_[df_.columns.difference(self.columns_exception)]
        df_.columns = ['MEDIA_'+column_name if column_name not in keys else column_name for column_name in df_.columns]
        return dataframe_main.merge(df_, how='left', on=keys).copy()
    
class ContagemGroupBy():
    def __init__(self, new_column_name):
        self.new_column_name = new_column_name
        
    def transform(self, keys, dataframe_main, dataframe_static):
        df_ = dataframe_static.groupby(keys).size().reset_index(name=self.new_column_name)
        return dataframe_main.merge(df_, how='left', on=keys).copy()
    
class FeatureSelectionLSTM():
    def __init__(self, columns_to_add):
        self.columns_to_add = columns_to_add
    
    def transform(self, keys, dataframe_main, dataframe_static):
        dataframe_main = dataframe_main.merge(dataframe_static[keys + self.columns_to_add].drop_duplicates(), 
                                               how='left',
                                               on=keys).copy()
        return dataframe_main.iloc[:,3:].copy()
    
class Resize():
    def __init__(self, quantidade_semestre=4):
        self.quantidade_semestre = quantidade_semestre
    
    def transform(self, dataframe):
        return np.reshape(dataframe.values, (-1, self.quantidade_semestre, len(dataframe.columns))).copy()
    
###################################################################################

class TransformerCNN():
    def __init__(self, window_size):
        self.window_size = window_size
        
    def transform(self, time_series_dataframe):     
        num_registros = time_series_dataframe['ID_CURSO_ALUNO'].nunique()
        data_list = []
        for elem in time_series_dataframe.groupby(['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX']):
            diff_size = int(self.window_size - len(elem[1].index))
            row = elem[1].loc[:,elem[1].columns.difference(['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX'])]
            if diff_size != 0 :
                data_list.append( np.append(row.values, np.zeros((diff_size, len(row.columns))), axis=0) )
            else:
                data_list.append( row.values )
        return np.array(data_list).reshape((int(num_registros),-1,int(self.window_size),int(len(row.columns))))[:,:,:,:].copy()


# In[4]:


time_series_columns = ['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX', 'NEW_CREDITOS',
                        'MEDIA_FINAL', 'NEW_ANO_DISCIPLINA', 'NEW_CH_DISCIPLINA',
                       'NEW_NUM_FALTAS', 'NEW_SEMESTRE_DISCIPLINA', 'NEW_SITUACAO_DISCIPLINA_APROVADO',
                       'NEW_SITUACAO_DISCIPLINA_APROVEITAMENTO', 'NEW_SITUACAO_DISCIPLINA_NÃO APROVADO',
                       'NEW_SITUACAO_DISCIPLINA_OUTRAS',
                       'NEW_SITUACAO_DISCIPLINA_NÃO APROVADO POR FREQUENCIA']

static_columns = ['ID_CURSO_ALUNO',
                  'ANO_INGRESSO', 'CH_CURSO', 'COD_CURSO', 'COTISTA',
                  'NEW_FORMA_INGRESSO', 'NEW_NACIONALIADE', 'NEW_NATURALIDADE_NATURALIDADE_4',
                  'NEW_NATURALIDADE_NATURALIDADE_5', 'NEW_NATURALIDADE_OUTRAS',
                  'NEW_UF_NATURALIDADE_OUTRAS', 'NEW_UF_NATURALIDADE_UF_NATURALIDADE_0',
                  'NEW_UF_NATURALIDADE_UF_NATURALIDADE_1',
                  'PERIODO_INGRESSO', 'PLANO_ESTUDO', 'LABEL']

class Filter():
    def __init__(self, time_series_columns, static_columns, quantidade_semestre=4, window_size=35):
        self.time_series_columns = time_series_columns
        self.static_columns = static_columns
        self.quantidade_semestre = quantidade_semestre
        self.window_size = window_size

    def transform(self, dataframe):
        temp_dataframe = dataframe[['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX']]                    .sort_values(['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX'], ascending=[True, False, False])                    .drop_duplicates()                    .copy()
        temp_dataframe['quantidade_semestre'] = temp_dataframe.groupby(['ID_CURSO_ALUNO']).cumcount() + 1
        temp_dataframe = pd.merge(dataframe,
                                 temp_dataframe[(temp_dataframe['ID_CURSO_ALUNO'].isin(temp_dataframe[temp_dataframe['quantidade_semestre']==self.quantidade_semestre]['ID_CURSO_ALUNO'].unique()))&
                                          (temp_dataframe['quantidade_semestre']<=self.quantidade_semestre)],
                                 on=['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX'],
                                 how='inner')\
                            .sort_values(['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX'])\
                            .copy()
        
        df_temp = temp_dataframe.groupby(['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX']).size().reset_index(name='window_size')
        temp_dataframe = temp_dataframe[~temp_dataframe['ID_CURSO_ALUNO'].isin(df_temp[df_temp['window_size'] > self.window_size]['ID_CURSO_ALUNO'].unique())]
        
        time_series_dataframe, static_dataframe = temp_dataframe.loc[:, temp_dataframe.columns.isin(self.time_series_columns)], temp_dataframe.loc[:, temp_dataframe.columns.isin(self.static_columns)]

        # ----- STATIC SERIES -----
        data_numpy_static = static_dataframe.drop_duplicates()[static_dataframe.columns.difference(['LABEL','ID_CURSO_ALUNO'])].values
        label_numpy_static = static_dataframe.drop_duplicates()[['LABEL']].values
        
        return time_series_dataframe.copy(), data_numpy_static.copy(), label_numpy_static.copy()


# In[5]:


WINDOW_SIZE = 35
QUANTIDADE_SEMESTRE = 4

semestre_disciplina_remover_alunos = RemoverPorMap('SEMESTRE_DISCIPLINA', map_semestre_disciplina)
forma_evasao_remover_alunos = RemoverPorMap('FORMA_EVASAO', map_forma_evasao)

# REMOÇÃO DE DADO INCOERENTE
ano_disciplina_remover = Remover('ANO_DISCIPLINA', [2106])

# DUPLICAÇÃO DE COLUNAS PARA INDEXAÇÃO DOS DADOS
ano_disciplina_duplicate = FeatureDuplicate('ANO_DISCIPLINA', 'NEW_ANO_DISCIPLINA')

# GENERALIZAÇÃO
forma_evasao_map_transformer = MapTransformer('FORMA_EVASAO', 'LABEL', map_forma_evasao)
semestre_disciplina_map_transformer = MapTransformer('SEMESTRE_DISCIPLINA', 'NEW_SEMESTRE_DISCIPLINA', map_semestre_disciplina)
forma_ingresso_map_transformer = MapTransformer('FORMA_INGRESSO', 'NEW_FORMA_INGRESSO', map_forma_ingresso, True)
situacao_disciplina_map_transformer = MapTransformer('SITUACAO_DISCIPLINA', 'NEW_SITUACAO_DISCIPLINA', map_situacao_disciplina, True)

semestre_disciplina_duplicate = FeatureDuplicate('NEW_SEMESTRE_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX')

# VALORES RAROS
nacionalidade_map_transformer = MapTransformer('NACIONALIADE', 'NEW_NACIONALIADE', map_nacionalidade, True)
naturalidade_map_transformer = MapTransformer('NATURALIDADE', 'NEW_NATURALIDADE', map_naturalidade, True)
uf_naturalidade_map_transformer = MapTransformer('UF_NATURALIDADE', 'NEW_UF_NATURALIDADE', map_uf_naturalidade, True)


# WINSORIZATION
creditos_winsorization = Winsorization('CREDITOS', 'NEW_CREDITOS')
ch_disciplina_winsorization = Winsorization('CH_DISCIPLINA', 'NEW_CH_DISCIPLINA')
num_faltas_winsorization = Winsorization('NUM_FALTAS', 'NEW_NUM_FALTAS')
# PREENCHER COM MÉDIA
media_final_fill_mean = FillMean('MEDIA_FINAL')
creditos_fill_mean = FillMean('NEW_CREDITOS')
ch_disciplina_fill_mean = FillMean('NEW_CH_DISCIPLINA')
num_faltas_fill_mean = FillMean('NEW_NUM_FALTAS')
ch_curso_fill_mean = FillMean('CH_CURSO')

label_encoder_cod_curso = LabelEncoder('COD_CURSO', {'info_COD_CURSO_0':0, 'info_COD_CURSO_1':1})
label_encoder_plano_estudo = LabelEncoder('PLANO_ESTUDO', {'info_PLANO_ESTUDO_0':0, 'info_PLANO_ESTUDO_1':1})
label_encoder_costista = LabelEncoder('COTISTA', {'info_COTISTA_0':0, 'info_COTISTA_1':1})
label_encoder_periodo_ingresso = LabelEncoder('PERIODO_INGRESSO', {'1º Semestre':0, '2º Semestre':1})
label_encoder_semestre_disciplina = LabelEncoder('NEW_SEMESTRE_DISCIPLINA', {'1º Semestre':0, '2º Semestre':1})
label_encoder_forma_ingresso = LabelEncoder('NEW_FORMA_INGRESSO', {'VESTIBULAR_OU_SISU':0, 'OUTRAS':1})
label_encoder_nacionalidade = LabelEncoder('NEW_NACIONALIADE', {'info_NACIONALIADE_0':0, 'OUTRAS':1})

scaler = StandardScaler(['ANO_INGRESSO', 'CH_CURSO',
                       'COD_CURSO', 'PLANO_ESTUDO', 'COTISTA', 'PERIODO_INGRESSO',
                        'NEW_SEMESTRE_DISCIPLINA', 'NEW_FORMA_INGRESSO', 'NEW_NACIONALIADE'])

one_hot_encoder = OneHotEncoder(['NEW_NATURALIDADE', 'NEW_UF_NATURALIDADE', 'NEW_SITUACAO_DISCIPLINA'])
feature_selection = FeatureSelection(['CH_DISCIPLINA', 'CREDITOS', 'EMPREGO_SALARIO', 'EMPREGO_SITUACAO',
                                     'FORMA_EVASAO', 'FORMA_INGRESSO', 'GASTO_MORADIA',
                                     'MORADIA_SITUACAO', 'NACIONALIADE', 'NATURALIDADE',
                                     'NOME_CURSO', 'NOME_DISCIPLINA', 'NUM_FALTAS', 'NUM_MAX_PERIODOS',
                                     'NUM_PERIODOS_SUGERIDO', 'PERIODO_ALUNO', 'RENDA_PER_CAPITA_AUFERIDA',
                                     'SITUACAO_DISCIPLINA', 'TIPO_AUXILIO', 'SEMESTRE_DISCIPLINA',
                                     'TIPO_INSTUICAO_SEGUNDO_GRAU', 'UF_NATURALIDADE',
                                     'NEW_NATURALIDADE', 'NEW_SITUACAO_DISCIPLINA',
                                     'NEW_UF_NATURALIDADE', 'COD_DISCIPLINA'],
                                    True)

filter_function = Filter(time_series_columns, static_columns, QUANTIDADE_SEMESTRE, WINDOW_SIZE)

#################### LSTM ####################
soma_group_by = SomaGroupBy(['NEW_ANO_DISCIPLINA', 'NEW_SEMESTRE_DISCIPLINA'])
desvio_group_by = DesvioGroupBy(['NEW_ANO_DISCIPLINA', 
                                 'NEW_SEMESTRE_DISCIPLINA',
                                 'NEW_SITUACAO_DISCIPLINA_APROVADO', 
                                 'NEW_SITUACAO_DISCIPLINA_APROVEITAMENTO', 
                                 'NEW_SITUACAO_DISCIPLINA_NÃO APROVADO',
                                 'NEW_SITUACAO_DISCIPLINA_NÃO APROVADO POR FREQUENCIA'])
media_group_by = MediaGroupBy(['NEW_ANO_DISCIPLINA', 'NEW_SEMESTRE_DISCIPLINA'])
contagem_group_by = ContagemGroupBy('QUANTIDADE_DISCIPLINAS')
feature_selection_lstm = FeatureSelectionLSTM(['NEW_ANO_DISCIPLINA', 'NEW_SEMESTRE_DISCIPLINA'])
scaler_lstm = StandardScaler(['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX'], True)
resize = Resize(QUANTIDADE_SEMESTRE)

#################### CNN ####################
scaler_cnn = StandardScaler(['NEW_ANO_DISCIPLINA', 'MEDIA_FINAL', 'NEW_CREDITOS', 'NEW_CH_DISCIPLINA', 'NEW_NUM_FALTAS'])
transformer_cnn = TransformerCNN(WINDOW_SIZE)


# In[48]:


pipe_default = Pipeline([ano_disciplina_remover,
                     ano_disciplina_duplicate,
                     semestre_disciplina_remover_alunos,
                     forma_evasao_remover_alunos,

                     forma_evasao_map_transformer, 
                     forma_ingresso_map_transformer,
                     nacionalidade_map_transformer,
                     naturalidade_map_transformer,
                     uf_naturalidade_map_transformer,
                     semestre_disciplina_map_transformer,
                     situacao_disciplina_map_transformer,

                     semestre_disciplina_duplicate,

                     creditos_winsorization,
                     ch_disciplina_winsorization,
                     num_faltas_winsorization,

                     media_final_fill_mean,
                     creditos_fill_mean,
                     ch_disciplina_fill_mean,
                     num_faltas_fill_mean,
                     ch_curso_fill_mean,
                         
                     label_encoder_cod_curso, 
                     label_encoder_plano_estudo,
                     label_encoder_costista,
                     label_encoder_periodo_ingresso,
                     label_encoder_semestre_disciplina,
                     label_encoder_forma_ingresso,
                     label_encoder_nacionalidade,
                     
                     scaler,
                         
                     one_hot_encoder,

                     feature_selection
                    ], filter_function)

pipe_lstm = PipelineLSTM([soma_group_by,
                          desvio_group_by,
                          media_group_by,
                          contagem_group_by,
                          feature_selection_lstm,
                          scaler_lstm
                         ],
                         ['ID_CURSO_ALUNO', 'ANO_DISCIPLINA', 'SEMESTRE_DISCIPLINA_INDEX'],
                         resize)

pipe_cnn = Pipeline([scaler_cnn,
                    transformer_cnn])

