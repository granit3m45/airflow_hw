import glob
import json
import os
import dill
from datetime import datetime as dt
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')
# path = os.path.expanduser('~/airflow_hw')
def predict():
    models = sorted(os.listdir(f'{path}/data/models/'))
    with open(f'{path}/data/models/{models[-1]}', 'rb') as file:
        model = dill.load(file)
    df_preds = pd.DataFrame(columns=['car_id', 'predict'])
    for file in glob.glob(f'{path}/data/test/*.json'):
        with open(file) as fnfile:
            form = json.load(fnfile)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            X = {'car_id': df.id, 'predict': y }
            df_pred = pd.DataFrame(X)
            df_preds = pd.concat([df_preds, df_pred], ignore_index=True)
    df_preds.to_csv(f'{path}/data/predictions/preds_{dt.now().strftime("%Y%m%d%H%M")}.csv', index=False)
    print(df_preds)


if __name__ == '__main__':
    predict()
