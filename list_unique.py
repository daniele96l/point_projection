import pandas as pd
import numpy as np
data = pd.read_csv('rilevamenti.txt', sep = ',', header = None)

data = np.stack(data[4].values)

name = [x[:-6] for x in data]


name = pd.DataFrame(name)
signs = name.drop_duplicates()
signs[0] = signs[0].str.replace("'",'')

print(signs)
