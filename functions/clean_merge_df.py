#import libraries
import pandas as pd
import numpy as np
import os.path
#from os.path import dirname


#load databases: covid and mobility
url_covid='https://github.com/owid/covid-19-data/blob/9de645e6646f7cd91d389c6263c10f306c6bb201/public/data/owid-covid-data.csv?raw=true'
url_mob='https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'

df_covid=pd.read_csv(url_covid, index_col=0)
df_mob=pd.read_csv(url_mob)

df_covid=df_covid.reset_index()

#rename columns of the mobility categories for better future visualization
df_mob=df_mob.rename(columns={'retail_and_recreation_percent_change_from_baseline':'retail_and_recreation',
       'grocery_and_pharmacy_percent_change_from_baseline':'grocery_and_pharmacy',
       'parks_percent_change_from_baseline': 'parks',
       'transit_stations_percent_change_from_baseline': 'transit_stations',
       'workplaces_percent_change_from_baseline': 'workplaces',
       'residential_percent_change_from_baseline': 'residential'})

#apply format to date column
df_covid['date']=pd.to_datetime(df_covid['date'],dayfirst=False)
df_mob['date']=pd.to_datetime(df_mob['date'],dayfirst=False)

# Since the purpose of this analysis is to focus in the cases due to mobility, we can remove some columns that will not be used in the analysis
df_covid_filtered=df_covid.drop(columns=[ 'reproduction_rate', 'icu_patients',
       'icu_patients_per_million', 'hosp_patients',
       'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
       'weekly_hosp_admissions_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand',
       'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
       'positive_rate', 'tests_per_case', 'tests_units',
       'population', 'population_density', 'median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy', 'human_development_index',
       'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
       'excess_mortality', 'excess_mortality_cumulative_per_million', 
        'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
       'people_fully_vaccinated_per_hundred', 'new_people_vaccinated_smoothed_per_hundred','total_boosters_per_hundred',]).copy()
df_mob_fil=df_mob.drop(columns=["metro_area","sub_region_2","iso_3166_2_code", "census_fips_code"])

#merge both df, for mobility only considering the general of the country
df_merged=pd.merge(df_mob_fil[df_mob_fil["sub_region_1"].isnull()],df_covid_filtered, how="inner", left_on=["date","country_region"], right_on=["date","location"])

#add column days
df_merged['day_of_week']=df_merged['date'].dt.dayofweek
df_merged['is_weekend']=np.where(df_merged['day_of_week']>=5,"Weekend","Weekday")

df_merged=df_merged.drop(columns=["location","sub_region_1"])


parent = os.path.dirname(os.getcwd())
path = parent + '\\data\\'

#convert to csv
df_merged.to_csv(path+"df_merged.csv")