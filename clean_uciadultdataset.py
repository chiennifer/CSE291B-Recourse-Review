import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import recourse as rs
from recourse.builder import RecourseBuilder
from recourse.builder import ActionSet

adultDF = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                     header=0, names =  
                     ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "prediction"])

# adultDF.head()

adultDF = adultDF[adultDF["workclass"]!=' ?']

adultDF = adultDF[adultDF["occupation"]!=' ?']

adultDF = (adultDF
					.assign(race_White=
                          lambda df:(df['race']==' White').astype(int))
                    .assign(race_AsianPacIslander=
                          lambda df:(df['race']==' Asian-Pac-Islander').astype(int))
                    .assign(race_AmerIndianEskimo=
                          lambda df:(df['race']==' Amer-Indian-Eskimo').astype(int))
                    .assign(race_Other=
                          lambda df:(df['race']==' Other').astype(int))
#                   .assign(race_Black=
#                          lambda df:(df['race']==' Black').astype(int))

					


					.assign(isMale=
                          lambda df:(df['sex']==' Male').astype(int))
                  	.drop(['sex', 'race'], axis=1))



