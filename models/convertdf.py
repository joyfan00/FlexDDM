import pandas as pd 

def convertToDF(tuple_data, participant_id):
    return pd.DataFrame({
        'id': [participant_id] * len(tuple_data[0]),
        'trial': tuple_data[0],
        'accuracy': tuple_data[1], # previously choice
        'rt': tuple_data[2],
        'congruency': tuple_data[3],
    })