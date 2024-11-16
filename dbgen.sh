#!/bin/bash

# Variables
DB_NAME="tpch10"
PG_USER="postgres"  
PG_HOST="127.0.0.1"      
PG_PORT="5432"           
DBGEN_PATH=$DSS_CONFIG
DATA_DIR=$DSS_PATH
SCHEMA_FILE="tpch_schema.sql"
CONSTRAINTS_FILE="tpch_constraints.sql"


echo "Checking if database $DB_NAME exists..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"
if [ $? -eq 0 ]; then
  echo "Database $DB_NAME deleted successfully (if it existed)."
else
  echo "Failed to delete database $DB_NAME. Please check for errors."
  exit 1
fi

echo "Running dbgen with scale factor 10..."
$DBGEN_PATH -vf -s 10
if [ $? -eq 0 ]; then
  echo "Data generation completed."
else
  echo "Failed to generate data with dbgen. Please check for errors."
  exit 1
fi

echo "Creating database $DB_NAME..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -d postgres -c "CREATE DATABASE $DB_NAME;"
if [ $? -eq 0 ]; then
  echo "Database $DB_NAME created successfully."
else
  echo "Failed to create database $DB_NAME. Please check for errors."
  exit 1
fi

echo "Setting up the database schema using $SCHEMA_FILE..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -d $DB_NAME -f $SCHEMA_FILE
if [ $? -eq 0 ]; then
  echo "Database schema setup completed."
else
  echo "Failed to set up schema. Please check for errors."
  exit 1
fi

echo "Copying .tbl files from $DATA_DIR into the database $DB_NAME..."
for file in $DATA_DIR/*.tbl; do
  table_name=$(basename "$file" .tbl) 
  echo "Importing data into table $table_name from $file..."
  psql -U $PG_USER -h $PG_HOST -p $PG_PORT -d $DB_NAME -c "\COPY $table_name FROM '$file' WITH (FORMAT csv, DELIMITER '|');"
done


echo "Applying constraints from $CONSTRAINTS_FILE..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -d $DB_NAME -f $CONSTRAINTS_FILE
if [ $? -eq 0 ]; then
  echo "Constraints applied successfully."
else
  echo "Failed to apply constraints. Please check for errors."
  exit 1
fi

echo "Checking if database ${DB_NAME}_sample exists..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME}_sample;"
if [ $? -eq 0 ]; then
  echo "Database ${DB_NAME}_sampledeleted successfully (if it existed)."
else
  echo "Failed to delete database ${DB_NAME}_sample. Please check for errors."
  exit 1
fi

echo "Creating database ${DB_NAME}_sample..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -d postgres -c "CREATE DATABASE ${DB_NAME}_sample;"
if [ $? -eq 0 ]; then
  echo "Database ${DB_NAME}_sample created successfully."
else
  echo "Failed to create database ${DB_NAME}_sample. Please check for errors."
  exit 1
fi


echo "Setup for database $DB_NAME completed successfully."
