import pandas as pd 

def convertToDF(tuple_data, participant_id):
    return pd.DataFrame({
        'id': [participant_id] * len(tuple_data[0]),
        'trial': tuple_data[0],
        'accuracy': tuple_data[1], # previously choice
        'rt': tuple_data[2],
        'congruency': tuple_data[3],
    })

def getRTData(path, id, congruency, rt, accuracy):
    """
    Gets the reaction time data from file. 

    @path: the path to the data 
    """
    data = pd.read_csv(path)
    data = pd.DataFrame({'id': data[id], 'congruency': data[congruency],'rt': [x for x in data[rt]], 'accuracy': data[accuracy]})
    data['congruency'] = ['congruent' if x == 1 else 'incongruent' for x in data['congruency']]
    return data