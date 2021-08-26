-- 1. Create a database called house_price_regression.
create database house_price_regression;
-- 2. Create a table house_price_data with the same columns as given in the csv file. 
-- Please make sure you use the correct data types for the columns. You can find the names of the 
-- headers for the table in the regression_data.xls file. Use the same column names as the names in 
-- the excel file. Please make sure you use the correct data types for each of the columns.

-- create table if not exists house_price_data (id INT UNIQUE NOT NULL, 'date' datetime default null, ...) doeas not work. Instead used Import Wizard

-- 3. Import the data from the csv file into the table. 
-- Before you import the data into the empty table, make sure that you have deleted the headers from the csv 
-- file. (in this case we have already deleted the header names from the csv files). 
-- To not modify the original data, if you want you can create a copy of the csv file as well. 
-- Note you might have to use the following queries to give permission to SQL to import data from csv files 
-- in bulk:  
SHOW VARIABLES LIKE 'local_infile'; -- This query would show you the status of the variable ‘local_infile’. If it is off, use the next command, otherwise you should be good to go
SET GLOBAL local_infile = 1;
-- import via Wizard

-- 4.Select all the data from table house_price_data to check if the data was imported correctly
use house_price_regression;
select * from house_price_data;

-- 5. Use the alter table command to drop the column date from the database, as we would not use 
-- it in the analysis with SQL. Select all the data from the table to verify if the command worked. 
-- Limit your returned results to 10.

ALTER TABLE house_price_data 
drop column date;

select * from house_price_data
limit 10;

-- 6. Use sql query to find how many rows of data you have.

select count(*) from house_price_data;
-- 21597 rows.

-- 7. Now we will try to find the unique values in some of the categorical columns:
	-- What are the unique values in the column bedrooms?
    select distinct bedrooms from house_price_data
    order by bedrooms;
	-- What are the unique values in the column bathrooms?
    select distinct bathrooms from house_price_data
    order by bathrooms;
	-- What are the unique values in the column floors?
    select distinct floors from house_price_data
    order by floors;
	-- What are the unique values in the column condition?
    select distinct house_price_data.condition from house_price_data
    order by house_price_data.condition;
	-- What are the unique values in the column grade?
    select distinct grade from house_price_data
    order by grade;

-- 8. Arrange the data in a decreasing order by the price of the house. Return only the IDs of the top 10 most expensive houses in your data.
select id from house_price_data
order by price desc
limit 10;

-- 9. What is the average price of all the properties in your data?
select round(avg(price),2) as average_price from house_price_data;
-- 540,296.57

-- 10. In this exercise we will use simple group by to check the properties of some of the categorical variables in our data

	-- What is the average price of the houses grouped by bedrooms? 
		-- The returned result should have only two columns, bedrooms and Average of the prices. 
		-- Use an alias to change the name of the second column.
    select bedrooms, round(avg(price),2) as avg_price from house_price_data
    group by bedrooms
    order by bedrooms;
	
	-- What is the average sqft_living of the houses grouped by bedrooms? 
		-- The returned result should have only two columns, bedrooms and Average of the sqft_living. 
		-- Use an alias to change the name of the second column.
	select bedrooms, avg(sqft_living) as avg_sqft from house_price_data
    group by bedrooms
    order by avg_sqft;

	-- What is the average price of the houses with a waterfront and without a waterfront? 
	-- The returned result should have only two columns, waterfront and Average of the prices. Use an alias to change the name of the second column.
    select waterfront, round(avg(price),2) as avg_price from house_price_data
    group by waterfront;

	-- Is there any correlation between the columns condition and grade? 
	-- You can analyse this by grouping the data by one of the variables and then aggregating 
	-- the results of the other column. Visually check if there is a positive correlation or negative 
	-- correlation or no correlation between the variables.
    
    select house_price_data.condition, avg(grade) from house_price_data
    group by house_price_data.condition
    order by house_price_data.condition;
    
    -- positive correlation, the higher the grade, the better the condition.

	-- You might also have to check the number of houses in each category (ie number of houses for a given 
	-- condition) to assess if that category is well represented in the dataset to include it in your 
	-- analysis. For eg. If the category is under-represented as compared to other categories, ignore 
	-- that category in this analysis
    
    -- choosing category: Number of bedrooms
    select bedrooms, count(bedrooms) as number_of_houses from house_price_data
    group by bedrooms
    order by number_of_houses;
    -- houses with 7 bedrooms and more are under-represented.

-- 11. One of the customers is only interested in the following houses:
	-- Number of bedrooms either 3 or 4
	-- Bathrooms more than 3
	-- One Floor
	-- No waterfront
	-- Condition should be 3 at least
	-- Grade should be 5 at least
	-- Price less than 300000
	-- For the rest of the things, they are not too concerned. Write a simple query to find what are the 
	-- options available for them?
    -- select count(id) from house_price_data
    select id, bedrooms, bathrooms, price, grade, house_price_data.condition, waterfront, floors from house_price_data
    where price < 300000 and grade >= 5 and house_price_data.condition >= 3 and waterfront = 0 and floors =1 and (bedrooms=4 or bedrooms=3) and bathrooms >3;
    -- 3128 options without bedroom and bathromm choice
    -- 2294 choices when selecting all requirements except number of bathrooms
    -- when selecting number of bathroom > 3 -> no options left to choose. 
    -- If he reduced the number of bathrooms to >2 he would have 308 options to choose from.
    
    
    -- select count(id) from house_price_data
    select id,floors from house_price_data
    where (bedrooms =3 or bedrooms =4) and bathrooms > 3 and price < 300000 and grade >=5 and house_price_data.condition >=3 and waterfront = 0; -- and floors=1;
    
    -- There are 8 houses that meet all conditions except floors = 1, they all have 2 floors. 
    -- If customer is willing to give this condition up he would have 8 choices to choose from.
    
-- 12. Your manager wants to find out the list of properties whose prices are twice more than the 
-- average of all the properties in the database. Write a query to show them the list of such properties.
-- You might need to use a sub query for this problem.

select id, price from house_price_data
where price > 2*(select avg(price) from house_price_data);

select count(id) from house_price_data
where price > 2*(select avg(price) from house_price_data);

-- avg price of all houses: 540,296.57
-- 1246 houses are two times pricier than the average price of all houses


-- 13. Since this is something that the senior management is regularly interested in, create a view 
-- called Houses_with_higher_than_double_average_price of the same query.

drop view if exists Houses_with_higher_than_double_average_price;

create view Houses_with_higher_than_double_average_price as
select id, price from house_price_data
where price > 2*(select avg(price) from house_price_data)
order by price desc;

select * from Houses_with_higher_than_double_average_price; 

-- 14. Most customers are interested in properties with three or four bedrooms. What is the difference 
-- in average prices of the properties with three and four bedrooms? In this case you can simply use a 
-- group by to check the prices for those particular houses

select bedrooms, round(avg(price),2) as avg_price 
from house_price_data
group by bedrooms
having bedrooms =3 or bedrooms =4;

-- 15. What are the different locations where properties are available in your database? 
-- (distinct zip codes)

select distinct(zipcode) from house_price_data
order by zipcode;
-- range of zipcodes from 98001 - 98199

-- 16.Show the list of all the properties that were renovated.

select id, yr_renovated 
from house_price_data 
where yr_renovated !=0
order by yr_renovated;

-- number of renovations per year
select yr_renovated, count(id) as number_of_renovations_in_year
from house_price_data 
where yr_renovated !=0
group by yr_renovated
order by yr_renovated desc;

-- 17. Provide the details of the property that is the 11th most expensive property in your database.
select *, rank() over (order by price desc) as ranka
from house_price_data
where ranka =11;
-- does not work. Only works with subquery:
select *
from (
    select *,
    rank() over(order by price desc) as rankb
    from house_price_data
) subquery
where rankb=11;


