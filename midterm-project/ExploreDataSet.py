###Dataset donated by
# J.Czerniak, H.Zarzycki, Application of rough sets in the presumptive
# diagnosis of urinary system diseases, Artifical Inteligence and Security
# in Computing Systems, ACS'2002 9th International Conference Proceedings,
# Kluwer Academic Publishers,2003, pp. 41-51


# Attribute Information:
#
# a1 Temperature of patient { 35C-42C }
# a2 Occurrence of nausea { yes, no }
# a3 Lumbar pain { yes, no }
# a4 Urine pushing (continuous need for urination) { yes, no }
# a5 Micturition pains { yes, no }
# a6 Burning of urethra, itch, swelling of urethra outlet { yes, no }
# d1 decision: Inflammation of urinary bladder { yes, no }
# d2 decision: Nephritis of renal pelvis origin { yes, no }


# The first step in data analysis is understanding the dataset. Visualization and statistical analysis
#plays a big role in understanding the data. Lets consider some options to read and understand our dataset

# In this dataset, out of 6 independent variables, we have
#   1 continuous variable  - a1 or temperature
#   4 binary/discrete variables  (a2 - a5)
#
# 2 dependent binary variables
#   d1 and d2
#
# Since the output variable is binary, this is a classification problem
#


import numpy as np
data = np.load("data/acute_inflammation.tsv", delimiter = ' ', dtype=[('temp',int),
                                                                            ('nausea',bool),
                                                                            ('lumbar_pain',bool),
                                                                            ('freq_urine',bool),
                                                                            ('micturition_pain',bool),
                                                                            ('burning',bool),
                                                                            ('d1_inflammation',bool),
                                                                            ('d2_nephritis',bool)])

data
