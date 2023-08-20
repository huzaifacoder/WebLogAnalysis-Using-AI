import pandas as pd
from datetime import datetime

# Sample data
data = pd.DataFrame({
    'Time': [
        '29/Nov/2017:06:59:02',
        '30/Dec/2018:12:34:56'
    ]
})


# Replace "/" with "-" and map months
def preprocess_time(time_str):
    time_str = time_str.replace('/', '-')
    datetime_obj = datetime.strptime(time_str, '%d-%b-%Y:%H:%M:%S')
    month_numeric = datetime_obj.month
    return datetime_obj, month_numeric

data[['Time', 'MonthNumeric']] = data['Time'].apply(preprocess_time).apply(pd.Series)

print(data)
