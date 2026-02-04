import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

#Classe para transformar colunas numéricas
class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler= [
          'Age', 'Height', 'Weight',
          'FCVC', 'NCP', 'CH2O',
          'FAF', 'TUE'
          ]):
        self.min_max_scaler = min_max_scaler

    def fit(self, df):
        return self

    def transform(self, df):
      if (set(self.min_max_scaler).issubset(df.columns)):
        min_max = MinMaxScaler()
        df[self.min_max_scaler] = min_max.fit_transform(df[self.min_max_scaler])
        return df

      else:
        print("Colunas não encontradas na classe MinMax")
        return df
      

#Classe para transformar colunas string
class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding = [
          'Gender', 'family_history', 'FAVC', 'CAEC',
          'SMOKE', 'SCC', 'CALC', 'MTRANS'
          ]):
        self.OneHotEncoding = OneHotEncoding
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Initialize OneHotEncoder here

    def fit(self, df, y=None):
        cols_to_encode = [col for col in self.OneHotEncoding if col in df.columns]
        if cols_to_encode:
            self.encoder.fit(df[cols_to_encode])
        else:
            print("Warning: No specified OneHotEncoding columns found in DataFrame for fitting.")
        return self

    def transform(self, df):
      df_copy = df.copy()
      cols_to_encode = [col for col in self.OneHotEncoding if col in df_copy.columns]

      if not cols_to_encode:
        print("Warning: No specified OneHotEncoding columns found in DataFrame for transforming. Returning original DataFrame.")
        return df_copy

      one_hot_encoded_array = self.encoder.transform(df_copy[cols_to_encode])
      feature_names = self.encoder.get_feature_names_out(cols_to_encode)
      one_hot_df = pd.DataFrame(one_hot_encoded_array, columns=feature_names, index=df_copy.index)

      df_copy = df_copy.drop(columns=cols_to_encode)
      df_final = pd.concat([df_copy, one_hot_df], axis=1)

      return df_final
    

# Classe para transformar a coluna de obesidade de forma ordenada
class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_encoder= ['Obesity']):
        self.ordinal_encoder = ordinal_encoder

    def fit(self, df):
        return self

    def transform(self, df):
      if (set(self.ordinal_encoder).issubset(df.columns)):
        cols = [
            c for c in self.ordinal_encoder
            if c in df.columns and df[c].dtype == 'object'
        ]

        if cols:
            enc = OrdinalEncoder()
            df[cols] = enc.fit_transform(df[cols])

        return df

      else:
        print("Colunas não encontradas na classe OrdinalFeature")
        return df
      

#Inicialização da Pipeline de Pré-Processamento
preprocess_pipeline = Pipeline([
    ('min_max_scaler', MinMax()),
    ('one_hot_encoding', OneHotEncodingNames())
])