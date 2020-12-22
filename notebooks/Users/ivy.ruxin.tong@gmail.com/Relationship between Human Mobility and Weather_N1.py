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
# MAGIC - ***Dataset - Weather_CBG_2019 records daily weather parameters by census block group in the United States. There're 7 variables*** 
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
# MAGIC - ***Dataset - Social Distancing Metrics v2.1 is a product of Safegraph which aggregately summarizes daily views of USA foot-traffic between census block groups. There're 23 variables. For this analysis, I mainly use :***
# MAGIC 
# MAGIC      *description based on Safegraph documentation*
# MAGIC   
# MAGIC | Variable   | Description  | Unit | 
# MAGIC |-----------------  |---------------|------|
# MAGIC | origin_census_block_group    | 12-digit FIPS code for the Census Block Group |  | 
# MAGIC | distance_traveled_from_home  | Median distance traveled from the geohash-7 of the home by the devices measured within the time period. All distance with 0 has been excluded. | meter |
# MAGIC | date_range_start  | Start time for measurement | YYYY-MM-DDTHH:mm:SS±hh:mm |
# MAGIC | date_range_end  | End time for measurement  | YYYY-MM-DDTHH:mm:SS±hh:mm|
# MAGIC | device_count | Total number of devices seen during the date range | count  |
# MAGIC | median_home_dwell_time  | Median dwell time at home during the time period |  min |  
# MAGIC | completely_home_device_count  | Number of device devices do not leave the house during the time period | count  |   
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC - ***Additional variables created for the analysis ***
# MAGIC 
# MAGIC 
# MAGIC | Variable   | Description  | Unit | 
# MAGIC |-----------------  |---------------|------|
# MAGIC | month   | month of each year | 1-12 | 
# MAGIC | day  | day of each year | 1-31 |
# MAGIC | weekday  | day of each week | 0-6 (0 for Mon) |
# MAGIC | ratio of not leaving | completely_home_device_count / device_count | ratio between 0 and 1 |
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
# MAGIC 
# MAGIC 
# MAGIC ####**Notebook** : There are two notebooks for this project:
# MAGIC 
# MAGIC Notebook1 details the data analysis for 2019 weather dataset, Jan social distancing dataset, and initial Jan modeling (mainly in Python) [Current]   
# MAGIC 
# MAGIC Notebook2 idetails 2019 whole year modeling (mainly in Spark)

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
  .csv("/mnt/weathercbg/weather_cbg_2019.csv")
 )

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
g = sns.pairplot(variables)
g.fig.set_size_inches(15,10)

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

# sd_data_jan.to_csv("/dbfs/social_distance_data/jan.csv",index= False)  # save it for next time use

# COMMAND ----------

sd_data_jan = pd.read_csv("/dbfs/social_distance_data/jan.csv")
sd_data_jan.head()

# COMMAND ----------

sd_data_jan.shape

# COMMAND ----------

sd_data_jan[sd_data_jan.date_time > '1 days 00:00:00']

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

merge_Jan.info()

# COMMAND ----------

merge_Jan[merge_Jan.geoid==10730059033].head()

# COMMAND ----------

merge_Jan.describe()

# COMMAND ----------

# no missing value
merge_Jan.isna().sum()

# COMMAND ----------

corr = merge_Jan.iloc[:,np.r_[2,4,7:18]].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask,0)] = True
print('Correlation Matrix : ')
sns.heatmap(corr, mask = mask,square = True, annot = False , cmap="YlGnBu")

/

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,10)

# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 3
# MAGIC    - Response Visualization - 1
# MAGIC      - distance_traveled_from_home -  highly skewed

# COMMAND ----------

# # understand predictor  = distance traveled, highly right skewed
plt.hist(merge_Jan.distance_traveled_from_home, bins = range(0, 1000000,500))
plt.show()

# COMMAND ----------

# # understand predictor  = distance traveled, highly right skewed
plt.hist(merge_Jan.distance_traveled_from_home, bins = range(0, 1000000,500), log = True)
plt.show()

# COMMAND ----------

merge_Jan.distance_traveled_from_home[merge_Jan.distance_traveled_from_home > 500000].count() # 803
merge_Jan.distance_traveled_from_home[merge_Jan.distance_traveled_from_home > 100000].count() #5133
# how many 0 distance, this is important for later transformation, 18743 , not significant
merge_Jan.distance_traveled_from_home[merge_Jan.distance_traveled_from_home == 0].count()  # 18743
merge_Jan.distance_traveled_from_home[merge_Jan.distance_traveled_from_home > 10000].count() / merge_Jan.distance_traveled_from_home.count()

# COMMAND ----------

# Try a cut off of 50000 meters
plt.hist(merge_Jan.distance_traveled_from_home, bins = range(0, 500000,500), log = True)
plt.show()

# COMMAND ----------

# Try a cut off of 10000 meters
plt.hist(merge_Jan.distance_traveled_from_home, bins = range(0, 10000,500), log = True)
plt.show()

# COMMAND ----------

merge_Jan.columns

# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 3
# MAGIC    - Response Visualization 2
# MAGIC      - mean_home_dwell_home

# COMMAND ----------

plt.hist(merge_Jan.mean_home_dwell_time, bins = range(0, 1500,30))
plt.show()# 

# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 3
# MAGIC    - Response Visualization 3
# MAGIC      - ratio of not leaving

# COMMAND ----------

plt.hist(merge_Jan.ratio_not_leaving)
plt.show()# 

# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 4
# MAGIC    - Data Preposessing

# COMMAND ----------

# merge_Jan_v1 = merge_Jan.iloc[:,[2,11,12,13,14,15,16,17]]
merge_Jan_v1 = merge_Jan[merge_Jan.columns[2,11,12:18]]t
merge_Jan_v1 = merge_Jan.iloc[:,np.r_[2,11,12:18]]
merge_Jan_v1[merge_Jan_v1.tmin<0].count() # 421488, around 6%


merge_Jan_v1['tmin'] = merge_Jan_v1['tmin']+46
merge_Jan_v1['tmax'] = merge_Jan_v1['tmax']+46

def bottom_out(x):
  if x < 1:
    return 1
  return x

merge_Jan_v1['distance_traveled_from_home'] = merge_Jan_v1['distance_traveled_from_home'].map(bottom_out)
merge_Jan_v1['precip'] = merge_Jan_v1['precip'].map(bottom_out)

merge_Jan_v1.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 5
# MAGIC    - Visualization of response(mean_distance_traveled_from_home) with features
# MAGIC      - this response variable is not optimal 

# COMMAND ----------

merge_Jan_1 = merge_Jan[merge_Jan.distance_traveled_from_home < 500000]

# COMMAND ----------

# take a sample for visualization
merge_Jan_1_sample = merge_Jan_1.sample(frac=0.05)

fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_1_sample.groupby(['dayofweek']).median()['distance_traveled_from_home'].plot(ax=ax)
plt.ylabel('median distance travel')


# COMMAND ----------

# relationship between distance and exploratory variables

for x in merge_Jan_1_sample.columns[11:18]:
  plt.scatter(x=x, y='distance_traveled_from_home', data=merge_Jan_1_sample)
  plt.xlabel(x)
  plt.ylabel('distance')
  plt.show()


# COMMAND ----------

merge_Jan_2 = merge_Jan[merge_Jan.distance_traveled_from_home < 10000]

# COMMAND ----------

# take a sample for visualization
merge_Jan_2_sample = merge_Jan_2.sample(frac=0.05)

fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_2_sample.groupby(['dayofweek']).median()['distance_traveled_from_home'].plot(ax=ax)
plt.ylabel('median distance travel')


# COMMAND ----------

# relationship between distance and exploratory variables

for x in merge_Jan_2_sample.columns[11:18]:
  plt.scatter(x=x, y='distance_traveled_from_home', data=merge_Jan_2_sample)
  plt.xlabel(x)
  plt.ylabel('distance')
  plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 5
# MAGIC    - Visualization of response(mean_home_dwell_time) with features

# COMMAND ----------

# take a sample for visualization
merge_Jan_3_sample = merge_Jan.sample(frac=0.05)

fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_3_sample.groupby(['dayofweek']).median()['mean_home_dwell_time'].plot(ax=ax)
plt.ylabel('mean_home_dwell_time')

# COMMAND ----------

# relationship between distance and exploratory variables

for x in merge_Jan_3_sample.columns[11:18]:
  plt.scatter(x=x, y='mean_home_dwell_time', data=merge_Jan_3_sample)
  plt.xlabel(x)
  plt.ylabel('home dwell time')
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC  - Initial Modeling for Jan data only - Part 5
# MAGIC    - Visualization of response(ratio of not leaving) with features

# COMMAND ----------

# take a sample for visualization
fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_3_sample.groupby(['dayofweek']).median()['ratio_not_leaving'].plot(ax=ax)
plt.ylabel('ratio_not_leaving')

# COMMAND ----------

# relationship between distance and exploratory variables

for x in merge_Jan_3_sample.columns[11:18]:
  plt.scatter(x=x, y='ratio_not_leaving', data=merge_Jan_3_sample)
  plt.xlabel(x)
  plt.ylabel('ratio_not_leaving')
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC   - Initial Modeling for Jan data only - Part 6
# MAGIC    - Feature Engineering

# COMMAND ----------

y = merge_Jan_1['distance_traveled_from_home']

def bottom_out(x):
  if x < 1:
    return 1
  return x

y = y.map(bottom_out)
# y = y.apply(lamda x : 1 for x<1 else x)
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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# COMMAND ----------

# how to take care of this, like the scale, how many distance do we want cuz the accuracy score is so low

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