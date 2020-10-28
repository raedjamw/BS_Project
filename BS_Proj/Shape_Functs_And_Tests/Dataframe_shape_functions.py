# Lookback function
def lookback(dataset, timesteps):
    """
    The lookback function takes as input some number of seconds to look back in
    order to capture the past as features. Here the period is 60 seconds.
    this uses the shift method of pandas dataframes to shift all of
    the columns down one row and then append to the original dataset.
    """
    data = dataset
    for i in range(1, timesteps):
        # shift all of the columns down one row
        step_back = dataset.shift(i).reset_index()
        step_back.columns = ['index'] + [f'{column}_-{i}' for column in dataset.columns if column != 'index']
        # append shifted columns to the original dataset
        data = data.reset_index().merge(step_back, on='index', ).drop('index', axis=1)

    return data.dropna()
