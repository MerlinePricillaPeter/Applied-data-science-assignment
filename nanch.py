import pandas as pd

def C02_liquidfuel(filename):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(filename, header=2)

    # Set the "Country Name" column as the index
    df.set_index('Country Name', inplace=True)

    # Select columns from "1960" to "2021"
    df = df.loc[:, '1960':'2021']

    # Transpose the DataFrame so that the countries are columns and the years are rows
    df = df.transpose()

    # Load the CSV file into a Pandas DataFrame
    df_countries = pd.read_csv(filename, skiprows=4)

    # Set the "Country Name" column as the index
    df_countries.set_index('Country Name', inplace=True)

    # Select columns from "1960" to "2021"
    df_countries = df_countries.loc[:, '1960':'2021']
    
    return df, df_countries


#to output years as rows and countries as columns
years,countries = C02_liquidfuel('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv')

years

#countries as rows and years as columns
countries


countries.info()

countries.describe()

years.describe()

# Display info about the dataframe
years.info()


countries.corr()

#CO2 Emissions from Liquid Fuel Consumption (kt)
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv", skiprows=4)

# Set "Country Name" as the index
df = df.set_index("Country Name")

# Filter the data for the selected countries and years
countries = ["United States", "France", "China", "Japan"]
years = [str(i) for i in range(2011, 2016)]
df = df.loc[countries, years]

# Transpose the DataFrame
df = df.T

# Plot the line chart
df.plot(kind="line", figsize=(10, 5))

# Set the chart title and axis labels
plt.title("CO2 Emissions from Liquid Fuel Consumption (kt)")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions (kt)")

# Show the chart
plt.show()


#Fossil Fuel Energy Consumption (% of Total)
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_5354336.csv", skiprows=4)

# Set "Country Name" as the index
df.set_index("Country Name", inplace=True)

# Filter for the countries of interest
countries = ["United States", "France", "China", "Japan"]
df_countries = df.loc[countries]

# Extract the columns for the years of interest
years = ["2011", "2012", "2013", "2014", "2015"]
df_years = df_countries[years]

# Transpose the dataframe so that each country is a separate series
df_years = df_years.T

# Plot the multiple line graph
plt.plot(df_years)
plt.legend(df_years.columns)
plt.xlabel("Year")
plt.ylabel("Fossil Fuel Energy Consumption (% of Total)")
plt.title("Fossil Fuel Energy Consumption by Country")
plt.show()


#Alternative and Nuclear Energy Consumption
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("API_EG.USE.COMM.CL.ZS_DS2_en_csv_v2_4903166.csv", skiprows=4)

# Set the "Country Name" column as the index
df.set_index("Country Name", inplace=True)

# Select the columns for the years 2011 to 2015 for the countries United States, France, China, and Japan
cols = ["2011", "2012", "2013", "2014", "2015"]
df = df.loc[["United States", "France", "China", "Japan"], cols]

# Plot a bar graph
ax = df.plot(kind="bar")
ax.set_xlabel("Country")
ax.set_ylabel("% of Total Energy Use")
ax.set_title("Alternative and Nuclear Energy Consumption")

plt.show()


#Fossil Fuel Energy Consumption (% of total)
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv("API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_5354336.csv", skiprows=4)

# Select the rows for the specified countries and years
countries = ["United States", "France", "China", "Japan"]
years = ["2011", "2012", "2013", "2014", "2015"]
data = data.loc[data["Country Name"].isin(countries), ["Country Name"] + years]

# Set the country names as the index
data.set_index("Country Name", inplace=True)

# Create a bar plot
ax = data.plot(kind="bar")
ax.set_xlabel("Countries")
ax.set_ylabel("Fossil Fuel Energy Consumption (% of total)")
ax.set_xticklabels(countries, rotation=0)

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the four CSV files into DataFrames
fo = pd.read_csv("API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_5354336.csv", skiprows=4)
fa = pd.read_csv("API_AG.LND.FRST.K2_DS2_en_csv_v2_5346556.csv", skiprows=4)
an = pd.read_csv("API_EG.USE.COMM.CL.ZS_DS2_en_csv_v2_4903166.csv", skiprows=4)
epo = pd.read_csv("API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_5348180.csv", skiprows=4)

# Filter the DataFrames to only include data for China
fo = fo[fo["Country Name"] == "China"].transpose().iloc[4:]
fa = fa[fa["Country Name"] == "China"].transpose().iloc[4:]
an = an[an["Country Name"] == "China"].transpose().iloc[4:]
epo = epo[epo["Country Name"] == "China"].transpose().iloc[4:]

# Concatenate the four DataFrames along the columns
china = pd.concat([fo, fa, an, epo], axis=1)

# Rename the columns to match the names of the factors
china.columns = ["Fossil fuel energy consumption (% of total energy use)", "Forest area (% total)", "Alternative and nuclear energy", "Electricity production from oil sources"]

# Convert the data to numeric values
china = china.apply(pd.to_numeric)

# Select the data for the years 2011-2015
china = china.loc[["2011", "2012", "2013", "2014", "2015"]]

# Compute the correlation matrix
corr_matrix = china.corr()


for i in range(len(china.columns)):
    for j in range(len(china.columns)):
        text = plt.text(j, i, round(corr_matrix.iloc[i, j], 2),
                       ha="center", va="center", color="black")

# Generate the heatmap plot
plt.imshow(corr_matrix, cmap='coolwarm')
plt.xticks(range(len(china.columns)), china.columns, rotation=90)
plt.yticks(range(len(china.columns)), china.columns)
plt.colorbar()
plt.title("Correlation Heatmap for China (2011-2015)")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the four CSV files into DataFrames
fo = pd.read_csv("API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_5354336.csv", skiprows=4)
fa = pd.read_csv("API_AG.LND.FRST.K2_DS2_en_csv_v2_5346556.csv", skiprows=4)
an = pd.read_csv("API_EG.USE.COMM.CL.ZS_DS2_en_csv_v2_4903166.csv", skiprows=4)
epo = pd.read_csv("API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_5348180.csv", skiprows=4)

# Filter the DataFrames to only include data for France
fo = fo[fo["Country Name"] == "France"].transpose().iloc[4:]
fa = fa[fa["Country Name"] == "France"].transpose().iloc[4:]
an = an[an["Country Name"] == "France"].transpose().iloc[4:]
epo = epo[epo["Country Name"] == "France"].transpose().iloc[4:]

# Concatenate the four DataFrames along the columns
france = pd.concat([fo, fa, an, epo], axis=1)

# Rename the columns to match the names of the factors
france.columns = ["Fossil fuel energy consumption (% of total energy use)", "Forest area (% total)", "Alternative and nuclear energy", "Electricity production from oil sources"]

# Convert the data to numeric values
france = france.apply(pd.to_numeric)

# Select the data for the years 2011-2015
france = france.loc[["2011", "2012", "2013", "2014", "2015"]]

# Compute the correlation matrix
corr_matrix = france.corr()

# Generate the heatmap plot
plt.imshow(corr_matrix, cmap='coolwarm')
plt.xticks(range(len(france.columns)), france.columns, rotation=90)
plt.yticks(range(len(france.columns)), france.columns)
plt.colorbar()

# Put the correlation values in the resulting grids
for i in range(len(france.columns)):
    for j in range(len(france.columns)):
        plt.text(j, i, round(corr_matrix.iloc[i,j], 2), ha='center', va='center', color='white', fontsize=14)

plt.title("Correlation Heatmap for France (2011-2015)")
plt.show()
