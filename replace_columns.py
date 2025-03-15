import pandas as pd
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: python replace_columns.py original_file new_values_file output_file")
        sys.exit(1)

    original_file = sys.argv[1]
    new_values_file = sys.argv[2]
    output_file = sys.argv[3]

    # Read the original CSV (expected columns: x, y, speed, theta)
    df_orig = pd.read_csv(original_file, delimiter=';')
    # Read the new values CSV (expected columns: x_m, y_m with a leading '#' in header)
    df_new = pd.read_csv(new_values_file, delimiter=';')

    # Clean up column names: remove '#' and extra spaces
    df_new.columns = [col.strip().lstrip('#').strip() for col in df_new.columns]

    # Verify that the required columns exist after cleaning
    if 'x_m' not in df_new.columns or 'y_m' not in df_new.columns:
        print("Error: The new values CSV file does not contain the expected columns 'x_m' and 'y_m'")
        sys.exit(1)

    # Replace the columns in df_orig with values from df_new.
    # If the number of rows differs, only update up to the minimum number of rows.
    if len(df_orig) != len(df_new):
        print("Warning: The two CSV files have different numbers of rows.")
        min_len = min(len(df_orig), len(df_new))
        df_orig.loc[:min_len-1, 'x'] = df_new.loc[:min_len-1, 'x_m']
        df_orig.loc[:min_len-1, 'y'] = df_new.loc[:min_len-1, 'y_m']
    else:
        df_orig['x'] = df_new['x_m']
        df_orig['y'] = df_new['y_m']

    # Save the updated DataFrame back to a CSV file using semicolon as the delimiter
    df_orig.to_csv(output_file, index=False, sep=';')
    print(f"Replaced columns saved to {output_file}")

if __name__ == '__main__':
    main()
