import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
df = pd.read_csv('shoppers.csv')
y = df.iloc[:,-1]
features=['Administrative_Duration','ProductRelated_Duration','ExitRates','PageValues']
X = df.loc[:,features]

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scaler = StandardScaler()

sm = SMOTE(random_state=42)
smote_enn = SMOTEENN(smote = sm)
clf_log =LogisticRegression(C=0.000695193,class_weight='balanced',penalty='l1',solver='saga')


pipe_over_sample = Pipeline([('smote_enn', smote_enn),
                 ('scaler',scaler),
                ('clf_log', clf_log)])

regressor = clf_log.fit(X,y)
pickle.dump(regressor, open('model.pkl','wb'))
