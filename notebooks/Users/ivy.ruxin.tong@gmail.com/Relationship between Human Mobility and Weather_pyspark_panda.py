# Databricks notebook source
# MAGIC %md
# MAGIC ###Project Summary
# MAGIC <img src ='https://images.unsplash.com/photo-1548859047-1d15def63a14?ixlib=rb-1.2.1&ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=2700&q=80' width="800" height = "200">
# MAGIC ######Ivy(Ruxin) Tong | November 15th, 2020 | Image courtesy of Ivan Olenkevich

# COMMAND ----------

# MAGIC %md
# MAGIC ####**Objective** : Address the Challenge of Evaluation and Prediction of American’s Mobility under Extreme Weather Events
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ####**Datasets** :  Weather_CBG_2019 | Social Distancing Metrics v2.1 
# MAGIC ######***About the data***
# MAGIC - Dataset - Weather_CBG_2019 records daily weather parameters by census block group in the United States. There're 7 variables. 
# MAGIC 
# MAGIC | Variable   | Description  | Unit | 
# MAGIC |-----------------  |---------------|------|
# MAGIC | geoid    | census block group ID |  | 
# MAGIC | precip  | daily precipitation | mm|
# MAGIC | rmax | maximum daily relative humidity | % |
# MAGIC | rmin | mminimum daily relative humidity | % |
# MAGIC | srad | surface downwelling solar radiation | W/m^2 |
# MAGIC | tmax | maximum daily temperature | degress F |
# MAGIC | tmin | minimum daily temperature | degress F |
# MAGIC | wind_speed | wind speed | mph|
# MAGIC 
# MAGIC 
# MAGIC - Dataset - Social Distancing Metrics v2.1 is a product of Safegraph which aggregately summarizes daily views of USA foot-traffic between census block groups. There're 23 variables. For this analysis, I mainly use
# MAGIC   
# MAGIC   
# MAGIC | Variable   | Description  | Unit | 
# MAGIC |-----------------  |---------------|------|
# MAGIC | origin_census_block_group    | 12-digit FIPS code for the Census Block Group |  | 
# MAGIC | distance_traveled_from_home  | Median distance traveled from the geohash-7 of the home by the devices measured within the time period. All distance with 0 has been excluded. | m |
# MAGIC | distance_traveled_from_home  | Median distance traveled from the geohash-7 of the home by the devices measured within the time period. All distance with 0 has been excluded. | m |
# MAGIC 
# MAGIC 
# MAGIC   
# MAGIC   
# MAGIC ######***Acknowledgements*** : Social Distancing Metrics v2.1 is downloaded from Safegraph
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ####**Conclusion**:
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ####**Model and Result**:

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analysis
# MAGIC ###### Step 0 : Install and Import libraries

# COMMAND ----------

# MAGIC %run ./Packages_setup

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###### Step 1 : Data Preparation and Analysis for weather_cbg_2019
# MAGIC   - Connect AWS s3 bucket to Databricks and read the dataset

# COMMAND ----------

# # Secure access to S3 buckets using instance profiles
awsAccessKey = ""
# Encode the Secret Key to remove any "/" characters
secretKey = "".replace("/", "%2F")
awsBucketName = "weathercbg2019"
mountPoint = f"/mnt/weathercbg"

mountTarget = "s3a://{}:{}@{}".format(awsAccessKey, secretKey, awsBucketName)
dbutils.fs.mount(mountTarget, mountPoint)

# COMMAND ----------

dbutils.fs.ls("/mnt/weathercbg")

# COMMAND ----------

dbutils.fs.head("dbfs:/mnt/weathercbg/weather_cbg_2019.csv",1000)

# COMMAND ----------

# # #read weathercbgDF
weathercbgDF = (spark.read
  .option("delimiter", ",")
  .option("header", True)
  .option("inferSchema", True)
  .csv("/mnt/weathercbg/weather_cbg_2019.csv"))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC   - Transform and Explore the weather dataset

# COMMAND ----------

# #  Add three date-related variables
weathercbgDF = weathercbgDF.withColumn("month", F.month('date'))
weathercbgDF = weathercbgDF.withColumn("day", F.dayofmonth('date'))
weathercbgDF = weathercbgDF.withColumn("year", F.year('date'))
weathercbgDF.show(5)
# save to local folder so that we can access directly through databricks
weathercbgDF.write.format('com.databricks.spark.csv').save("/FileStore/weathercbgDF.csv",header = 'true',inferSchema = "true")

# COMMAND ----------

#   unmount s3
try:
  dbutils.fs.unmount(mountPoint)
except:
  print("{} already unmounted".format(mountPoint))

# COMMAND ----------

#decide to add a new variable after unmounting to s3
weathercbgDF = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("/FileStore/weathercbgDF.csv")
# weathercbgDF =weathercbgDF.withColumn('day_of_week',dayofweek(weathercbgDF.date))

def spark_df_shape(self):
    return (self.count(),len(self.columns)) 
pyspark.sql.dataframe.DataFrame.shape = spark_df_shape


print("weather cbg 2019 dataset")
weathercbgDF.printSchema()
print("shape:")
print(weathercbgDF.shape())

# COMMAND ----------

from pyspark.sql.functions import min, max

display(
  weathercbgDF.select(min("date"), max("date"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC   - Create summary table group by month and geoid & Basic visualization

# COMMAND ----------

#   create monthly summary table group by geoid, for visualization and faster operation
weather_cbg_month = (weathercbgDF.groupBy("geoid","month")
                            .agg(
                              mean("precip").alias("mean_precip"),
                              mean("rmax").alias("mean_rmax"),
                              mean("rmin").alias("mean_rmin"),
                              mean("srad").alias("mean_srad"),
                              mean("tmin").alias("mean_tmin"),
                              mean("tmax").alias("mean_tmax"),
                              mean("wind_speed").alias("mean_wind_speed"))
                             .sort(["geoid","month"]))

weather_cbg_month.show(5)

weather_cbg_month.write.format('com.databricks.spark.csv').save("/FileStore/weather_cbg_month.csv",header = 'true',inferSchema = 'true')

# COMMAND ----------

## Generate a sample for exploratory data analysis
weather_cbg_month = spark.read.format("csv").option("header", "true",).option("inferSchema", "true").load("/FileStore/weather_cbg_month.csv") 
weather_cbg_month.printSchema()

weather_cbg_2019_month_sample =weather_cbg_month.sample(0.01).toPandas()
weather_cbg_2019_month_sample.head()

# COMMAND ----------

# Understand distribution of Variables
variables = weather_cbg_2019_month_sample.drop(["geoid","month"], axis = 1)
fig,ax = plt.subplots(1,7, figsize= (30,5))

for i, variable in enumerate(variables):
  sns.distplot(variables[variable], ax=ax[i])

# COMMAND ----------

# these distributions are not on the same scale, standardize them
fix, ax = plt.subplots(1,7,figsize = (25,5))

for i, variable in enumerate(variables):
  original_variables = variables[variable]
  variable_scaled = (original_variables - original_variables.mean())/original_variables.std()
  sns.distplot(variable_scaled, ax=ax[i])
  ax[i].set_xlim(-2,2)

# COMMAND ----------

#correlation
sns.pairplot(variables)

# COMMAND ----------

corr = variables.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask,0)] = True
sns.heatmap(corr, mask = mask,square = True, annot = True)

# COMMAND ----------

# # scaled data by month. This is generally the US case, it varies by the region
for i, variable in enumerate(variables):
  original_variables = variables[variable]
  variable_scaled = (original_variables - original_variables.mean())/original_variables.std()
  sns.lineplot(x = "month", y=variable_scaled, data = weather_cbg_2019_month_sample,legend='brief', label=variable)

plt.ylabel("values")
plt.legend(bbox_to_anchor=(1.02, 1),borderaxespad=0.)
plt.title("mean weather parameters level by month")


# COMMAND ----------

# MAGIC %md
# MAGIC   - Prepare Jan file for initial modeling

# COMMAND ----------

# Prepare Jan file for initial modeling
weatherDF_Jan = weathercbgDF.where(col("month") == 1).toPandas()
weatherDF_Jan.head(10)

# COMMAND ----------

weatherDF_Jan.shape

# COMMAND ----------

weatherDF_Jan.info()

# COMMAND ----------

weatherDF_Jan.geoid.nunique()

# COMMAND ----------

weatherDF_Jan[weatherDF_Jan.geoid == 10730059033].head()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###### Step 2 : Data Preparation and Analysis for Social Distancing Metrics v2.1
# MAGIC   

# COMMAND ----------

# MAGIC %md 
# MAGIC   - Get data from Safegraph
# MAGIC     - Download this dataset from safegraph through cluster terminal to filepath 'cd databricks/driver/mnt/' -> Move them to '/dbfs/mnt#'

# COMMAND ----------

# use this command to move files so that I can directly access from databricks data UI
dbutils.fs.mv(r"file:/mnt/", r"dbfs:/social_distance_data/", True)

# COMMAND ----------

# MAGIC %fs ls /social_distance_data/social_distance/2019/

# COMMAND ----------

# MAGIC %fs ls /social_distance_data/social_distance/2019/01

# COMMAND ----------

# MAGIC %fs ls /social_distance_data/social_distance/2019/01/01

# COMMAND ----------

# MAGIC %md
# MAGIC   - Preparing SafeGraph social-distancing data
# MAGIC     - read one gzip file first to understand panda dataframe
# MAGIC     - read January data and prepare it for initial modeling, reason being that i wanna test the model out before scaling up the whole dataset
# MAGIC     - read all 2019 social-distancing dataset (approx 800M rows)

# COMMAND ----------

# use panda to read one file 
sd20190101 = pd.read_csv("/dbfs/social_distance_data/social_distance/2019/01/01/2019-01-01-social-distancing.csv.gz",compression = "gzip")
sd20190101.head() 

# COMMAND ----------

sd20190101.shape

# COMMAND ----------

sd20190101.info()

# COMMAND ----------

# MAGIC %md
# MAGIC   - Initial Modeling for Jan data only - Part 1
# MAGIC    - Prepare Jan social distancing dataset

# COMMAND ----------

# Prepare January file for initial modeling
Jan_files = glob.glob("/dbfs/social_distance_data/social_distance/2019/01/*/*")

li= []
for file in Jan_files:
    df = pd.read_csv(file, compression='gzip', usecols=['origin_census_block_group','date_range_start','date_range_end','distance_traveled_from_home','mean_home_dwell_time','completely_home_device_count','device_count'])
    li.append(df)

sd_data_jan = pd.concat(li, axis=0, ignore_index=True)
del li
sd_data_jan.head(10)

# COMMAND ----------

sd_data_jan['date_range_start'] = pd.to_datetime(sd_data_jan['date_range_start'],utc= True)
sd_data_jan['date_range_end'] = pd.to_datetime(sd_data_jan['date_range_end'],utc= True)
sd_data_jan['month'] = sd_data_jan['date_range_start'].dt.month
sd_data_jan['day'] = sd_data_jan['date_range_start'].dt.day
sd_data_jan['dayofweek'] = sd_data_jan['date_range_start'].dt.dayofweek
sd_data_jan['date_time'] = sd_data_jan['date_range_end']-sd_data_jan['date_range_start']
# sd_data_jan['date'] = [d.date() for d in sd_data_jan['date_range_start']]
sd_data_jan['ratio_not_leaving'] = round(sd_data_jan['completely_home_device_count']/sd_data_jan['device_count'],4)
sd_data_jan = sd_data_jan.drop(['date_range_start','date_range_end'],1)

# COMMAND ----------

# sd_data_jan.to_csv("/dbfs/social_distance_data/jan.csv",index= False)  # save it for next time use

# COMMAND ----------

# sd_data_jan = pd.read_csv("/dbfs/social_distance_data/jan.csv")
sd_data_jan.head()

# COMMAND ----------

sd_data_jan.shape

# COMMAND ----------

sd_data_jan.describe()

# COMMAND ----------

sd_data_jan[sd_data_jan.date_time > '1 days 00:00:00']

# COMMAND ----------

sd_data_jan.info()

# COMMAND ----------

sd_data_jan.origin_census_block_group.nunique()  # 219759
sd_data_jan.origin_census_block_group[~sd_data_jan.origin_census_block_group.isin(weatherDF_Jan.geoid)].nunique() # 4869

# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 2
# MAGIC    - Join Jan weather and Jan social distancing dataset together

# COMMAND ----------

# sd_data_jan = pd.read_csv("./social_distance_data/jan.csv")
merge_Jan = pd.merge(sd_data_jan.drop('date_time',1), weatherDF_Jan, how='inner', left_on=['origin_census_block_group','day','month'], right_on = ['geoid','day','month'])

# COMMAND ----------

merge_Jan.head()

# COMMAND ----------

merge_Jan[merge_Jan.geoid==10730059033].head()

# COMMAND ----------

merge_Jan.describe()

# COMMAND ----------

# # understand predictor  = distance traveled, highly right skewed
plt.hist(sd_data_jan.distance_traveled_from_home, bins = range(0, 1000000,500), log = True)
plt.show()

# COMMAND ----------

merge_Jan.distance_traveled_from_home[merge_Jan.distance_traveled_from_home > 500000].count()

# COMMAND ----------

# Try a cut off of 50000 meters
plt.hist(sd_data_jan.distance_traveled_from_home, bins = range(0, 500000,500), log = True)
plt.show()

# COMMAND ----------

merge_Jan.distance_traveled_from_home[merge_Jan.distance_traveled_from_home > 100000].count()

# COMMAND ----------

# Try a cut off of 50000 meters
plt.hist(sd_data_jan.distance_traveled_from_home, bins = range(0, 100000,500), log = True)
plt.show()

# COMMAND ----------

# how many 0 distance, this is important for later transformation, 18743 , not significant
merge_Jan.distance_traveled_from_home[merge_Jan.distance_traveled_from_home == 0].count()

# COMMAND ----------

merge_Jan = merge_Jan[merge_Jan.distance_traveled_from_home < 100000]

# COMMAND ----------

# take a sample for visualization
merge_Jan_sample = merge_Jan.sample(frac=0.3)

fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_sample.groupby(['dayofweek']).median()['distance_traveled_from_home'].plot(ax=ax)
plt.ylabel('median distance travel')


# COMMAND ----------

# relationship between distance and exploratory variables

for x in merge_Jan_sample.columns[11:18]:
  plt.scatter(x=x, y='distance_traveled_from_home', data=merge_Jan_sample)
  plt.xlabel(x)
  plt.ylabel('distance')
  plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC   - Initial Modeling for Jan data only - Part 3
# MAGIC    - Feature Engineering

# COMMAND ----------

merge_Jan_1 = merge_Jan.copy()
y = merge_Jan_1['distance_traveled_from_home']

def bottom_out(x):
  if x < 1:
    return 1
  return x

y = y.map(bottom_out)
y = stats.boxcox(y)[0]

sns.distplot(y)

# COMMAND ----------

# # lasso will drop the correlated variables, so we don't need to worry about it

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer


X = merge_Jan_1.iloc[:,np.r_[11:18,7]]
enc_col = X.columns[-1]
scal_col = X.columns[:-1]

encoded_X = pd.get_dummies(X.dayofweek)

scaler = StandardScaler()
# Scale, Lasso assumes Gaussian distribution
for i in scal_col:
  X.loc[:,i] = scaler.fit_transform(X[[i]])

X = np.concatenate([X.drop('dayofweek',1), encoded_X], axis=1)
X.shape 

# COMMAND ----------

# # lasso will drop the correlated variables, so we don't need to worry about it

# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

# merge_Jan_1 = merge_Jan.copy()
# X = merge_Jan_1.iloc[:,np.r_[12:19,7]]
# enc_col = X.columns[-1]
# scal_col = X.columns[:-1]

# pd.get_dummies(enc_col, prefix = enc_col)

# # Scale, Lasso assumes Gaussian distribution
# for i in scal_col:
#   X.loc[:,i] = scaler.fit_transform(X[[i]])
  
# encoded_X = pd.get_dummies(X.dayofweek)
# X = np.concatenate([X.drop('dayofweek',1), encoded_X], axis=1)
# X.shape 


# #one hot encoding the dayof week variable
# # ct_enc = ColumnTransformer([("enc", enc, enc_col)], remainder="passthrough")
# # ct_enc.fit_transform(X).shape

# # X_scale= merge_Jan_1[['precip','rmax','srad','tmax','wind_speed']]
# # X_encode =merge_Jan_1['dayofweek']
# # scaler = StandardScaler().fit(X_scale)
# # scaled_X = scaler.transform(X_scale)
# # encoded_X = pd.get_dummies(X_encode, prefix = X_encode)
# # X = np.concatenate([scaled_X, encoded_X], axis=1)


# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# COMMAND ----------

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

clf = linear_model.Ridge()

grid = GridSearchCV(estimator=clf, param_grid=dict(alpha=np.array([1,0.1,0.01,0.001,0.0001,0])))
grid.fit(X_train, y_train)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# param_grid = {   'n_estimators': [105, 120,150],
#                  'max_depth': [45, 50, 55]
#              }

# rf = RandomForestRegressor(random_state=2, max_features = 'sqrt', verbose = 3)
# grid_rf = GridSearchCV(rf, param_grid, cv=5)
# grid_rf.fit(X_train, y_train)

# rf = RandomForestRegressor(random_state=2, max_features = 'sqrt', verbose = 3)
# rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC   - Import all 2019 social distancing gzip files(approx. 800M rows)
# MAGIC   

# COMMAND ----------

# take too long to run, so switch to spark
# # prepare all 2019 social distance data
# import glob, os 
# all_files = glob.glob("/dbfs/social_distance_data/social_distance/2019/**/*.gz")

# li= []
# for file in all_files:
#     df = pd.read_csv(file, compression='gzip', usecols=['origin_census_block_group','date_range_start','date_range_end','distance_traveled_from_home','mean_home_dwell_time','completely_home_device_count','device_count'])
#     df['date_range_start'] = pd.to_datetime(df['date_range_start'],utc= True)
#     df['date_range_end'] = pd.to_datetime(df['date_range_end'],utc= True)
#     df['month'] = df['date_range_start'].dt.month
#     df['day'] = df['date_range_start'].dt.day
#     df['ratio_not_leaving'] = df['completely_home_device_count']/df['device_count']
#     df = df.drop(['date_range_start','date_range_end'],1)
    
#     li.append(df)

# sd_data = pd.concat(li, axis=0, ignore_index=True).to_csv('/dbfs/social_distance_data/sd_2019.csv',index=False,columns=cols)

# del all_files
# del li

# COMMAND ----------

#use wildcard to read all files into one. The main reason why i read all of them is that some columns might be used for later exploration
sd_data = spark.read.option("multiLine","true").option('escape',"\"").option('header',True).csv("dbfs:/social_distance_data/social_distance/2019/*/*/*.csv.gz")

display(sd_data)

# COMMAND ----------

# extract the files we need
sd_data_ = sd_data.select('origin_census_block_group','date_range_start','date_range_end','distance_traveled_from_home','mean_home_dwell_time','completely_home_device_count','device_count')


sd_data_ = sd_data_.withColumn("month", F.month('date_range_start'))
sd_data_ = sd_data_.withColumn("day", F.dayofmonth('date_range_start'))
sd_data_ = sd_data_.withColumn('day_of_week',dayofweek('date_range_start')-1)  # spark starts at 1, while pd at 0, be consistent
sd_data_ = sd_data_.withColumn('ratio_not_leaving', F.round(sd_data_['completely_home_device_count']/sd_data_['device_count'],4))
sd_data_v1 = sd_data_.drop(*['date_range_start','date_range_end'])

# COMMAND ----------

display(sd_data_v1)

# COMMAND ----------

sd_data_v1.write.format('com.databricks.spark.csv').save("/FileStore/sd_data_v1.csv",header = 'true',inferSchema = 'true')

# COMMAND ----------

sd_data_v1 = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("/FileStore/sd_data_v1.csv")

# COMMAND ----------

# 800K data
sd_data_v1_sample = sd_data_v1.sample(0.2).toPandas()

# COMMAND ----------

sd_data_v1_sample.describe()

# COMMAND ----------

# Understand response, highly right skewed
plt.hist(sd_data_v1_sample.distance_traveled_from_home, bins = range(0, 700000,500), log = True)
plt.show()

# COMMAND ----------

# Understand response
plt.hist(sd_data_v1_sample.distance_traveled_from_home, bins = range(0, 500000,500), log = True)
plt.show()

# COMMAND ----------

# only 2.4% of the sample datapoint is above 10000 meter every day
100*sd_data_v1_sample.distance_traveled_from_home[sd_data_v1_sample.distance_traveled_from_home > 10000].count()/sd_data_v1_sample.distance_traveled_from_home.count()


# COMMAND ----------

sd_data_v1_sample.groupby(['month','day'])['distance_traveled_from_home'].median().plot() 
plt.ylabel('median distance traveled')


# COMMAND ----------

fig,ax = plt.subplots(1,3, figsize= (30,5))

for i, x in enumerate(sd_data_v1_sample.columns[5:8]):
  sd_data_v1_sample.groupby([x]).median()['distance_traveled_from_home'].plot(ax=ax[i])
  plt.xlabel(x)

fig.suptitle('Median Distance Traveled', size = 20)
plt.show()

# COMMAND ----------

fig,ax = plt.subplots(1,3, figsize= (30,5)=)
for i, x in enumerate(sd_data_v1_sample.columns[5:8]):
  sd_data_v1_sample.groupby([x]).median()['ratio_not_leaving'].plot(ax=ax[i])
  plt.xlabel(x)

fig.suptitle('Median Ratio of Not Leaving The House', size=20)
plt.show()

# COMMAND ----------

headmap_vec = sd_data_v1_sample.groupby(['month','day_of_week'])['distance_traveled_from_home'].median()
matrix = headmap_vec.values.reshape((12,7))
matrix
p = sns.set(rc={'figure.figsize':(10,5)})
p = sns.heatmap(matrix.transpose(),cmap="YlGnBu")
p.set(xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
p.set(yticklabels=['Mon','Tue','Wed','Tr','Fri','Sat','Sun'],rotation=45) # 0 is Monday in pd


# COMMAND ----------

# here is just an idea to see if county code will be useful
sd_data_v1_sample['origin_census_block_group'] = sd_data_v1_sample['origin_census_block_group'].apply(str)
sd_data_v1_sample['origin_census_block_group'] = sd_data_v1_sample['origin_census_block_group'].str.zfill(12)
sd_data_v1_sample['origin_county_FIPS'] = sd_data_v1_sample['origin_census_block_group'].str[:5]   

# COMMAND ----------

sd_data_v1_sample.groupby(['origin_county_FIPS']).median()['distance_traveled_from_home'].plot()
plt.show()

# COMMAND ----------

sd_data_v1_sample['median_distance_by_county'] = sd_data_v1_sample.groupby("origin_county_FIPS")["distance_traveled_from_home"].transform('median')
sd_data_v1_sample

# COMMAND ----------

#  Note this part of the code was adapted from here -> https://plotly.com/python/choropleth-maps/

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

counties["features"][0]


# note this county is based on county Fips number which assumed to be consistent with the data source
import plotly.express as px
#  Note this part of the code was adapted from here -> https://plotly.com/python/choropleth-maps/
fig = px.choropleth(sd_data_v1_sample, geojson=counties, locations='origin_county_FIPS', color='median_distance_by_county',
                           color_continuous_scale= "viridis",
                           range_color=(0, 5),
                           scope="usa"
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()