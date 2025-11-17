# Life Expectancy by Country Dataset Analysis
The ["Country Life Expectancy"](https://www.kaggle.com/datasets/amirhosseinmirzaie/countries-life-expectancy/data) dataset (2848 x 18) provides population life expectancy and healthcare data for 178 countries from 2000 to 2015. \
This notebook will compare multiple regression models for predicting national life expectancy, including 
- LinearRegression, 
- RandomForestRegressor, 
- GradientBoostingRegressor, and 
- ExtraTreesRegressor.

### Dataset Description
Column descriptions: \
**Country:** \
Country under study	\
**Year:** \
year	\
**Population:** \
Status of the country's development	Status Population of country	\
**Hepatitis B:** \
Percentage of people finally one year old who were immunized against hepatitis B	\
**Measles:** \
The number of reported measles cases per 1000 people	\
**Polio:** \
Percentage of 1-year-olds immunized against polio	\
**Diphtheria:** \
Percentage of people finally one year old who were immunized against diphtheria	\
**HIV/AIDS:** \
The number of deaths caused by AIDS of the last 4-year-olds who were born alive per 1000 people	\
**infant deaths:** \
The number of infant deaths per 1000 people	\
**under-five deaths:** \
The number of deaths of people under 5 years old per 1000 people	\
**Total expenditure:** \
The ratio of government medical-health expenses to total government expenses in percentage	\
**GDP:** \
Gross domestic product	\
**BMI:** \
The average body mass index of the entire population of the country	\
**thinness 1-19 years:** \
Prevalence of thinness among people 19 years old in percentage	\
**Alcohol:** \
Liters of alcohol consumption among people over 15 years old	\
**Schooling:** \
The number of years that people study	\
**Life expectancy:** \
Country life expectancy	