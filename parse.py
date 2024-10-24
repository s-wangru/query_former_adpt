import re
import json

# Read the raw explain output
with open('tpch-kit/dbgen/output_file', 'r') as file:
    lines = file.readlines()

# Remove labels and merge all lines into one single string
clean_lines = []
for line in lines:
    # Ignore record labels like '-[ RECORD 1 ]---' and 'QUERY PLAN'
    if not re.match(r'^-\[ RECORD \d+ \]-', line) and not line.strip().startswith("QUERY PLAN") and line.strip().endswith("+"):
        clean_lines.append(line.replace('+', '').replace('|', '').strip())
    elif line.strip().startswith("QUERY PLAN"):
        clean_lines.append("\n")

# Join all lines and remove unnecessary characters (like '+' at the end of lines)
json_string = ''.join(clean_lines)

# Load the cleaned string into a Python dictionary to verify it's valid JSON
try:
    json_data = json.loads(json_string)
except json.JSONDecodeError as e:
    print("Error parsing JSON:", e)
    exit(1)

# Format the final output as requested
formatted_json_string = json.dumps(json_data, separators=(',', ':'))

# Example query to include (replace with your actual query if needed)


# Create the final output string
final_output = f'0,"{formatted_json_string}"'

# Write the final output to a file
with open('final_output.csv', 'w') as file:
    file.write(final_output)

print("Output has been saved to final_output.txt")
