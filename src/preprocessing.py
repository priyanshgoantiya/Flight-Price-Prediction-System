from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(num_base, cat_base):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_base),
        ('cat1', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_base)
    ])
    return preprocessor
