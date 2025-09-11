import pandas as pd
import sys

def calculate_metrics_from_csv(file_path):
    """
    Reads a CSV file, calculates metrics for each tenantId,
    and returns a DataFrame with the results.

    The CSV file must contain 'tenantId' and 'Status_v1' columns.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with the calculated metrics for each tenantId,
                          or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

    # Check if required columns exist in the DataFrame
    required_columns = ['tenantId', 'Status_v1']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The CSV must contain the following columns: {required_columns}")
        return None

    results = []
    # Group the DataFrame by tenantId to perform calculations for each tenant separately
    for tenant_id, group in df.groupby('tenantId'):
        # Count occurrences of each status type for the current tenant
        status_counts = group['Status_v1'].value_counts()

        gt_present_pr_present_match = status_counts.get('GT Present PR Present and match', 0)
        gt_present_pr_present_mismatch = status_counts.get('GT Present PR Present but mismatch', 0)
        gt_present_pr_absent = status_counts.get('GT Present PR Absent', 0)
        gt_absent_pr_present = status_counts.get('GT Absent PR Present', 0)

        # --- Metric Calculations ---

        # Coverage = (GT present and prediction present) / (Total fields whose GT is present)
        gt_present_total = gt_present_pr_present_match + gt_present_pr_present_mismatch + gt_present_pr_absent
        prediction_present_and_gt_present = gt_present_pr_present_match + gt_present_pr_present_mismatch
        coverage = prediction_present_and_gt_present / gt_present_total if gt_present_total > 0 else 0

        # Accuracy = (GT present and Prediction match) / (GT present and Prediction present)
        gt_present_and_prediction_present_total = gt_present_pr_present_match + gt_present_pr_present_mismatch
        accuracy = gt_present_pr_present_match / gt_present_and_prediction_present_total if gt_present_and_prediction_present_total > 0 else 0

        # Extra Fields = Count(GT absent PR present)
        extra_fields = gt_absent_pr_present

        results.append({
            "tenantId": tenant_id,
            "Coverage": f"{coverage:.2%}",
            "Accuracy": f"{accuracy:.2%}",
            "Extra Fields": extra_fields
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this variable to the path of your CSV file.
    csv_file_path = 'C:\\Users\\nitai.agarwal\\PycharmProjects\\PythonProject\\TIP_testing_pipeline\\outputs\\usg.cigna.834-proclaim\\consolidated_report.csv'

    print(f"Analyzing data from '{csv_file_path}'...")
    metrics_df = calculate_metrics_from_csv(csv_file_path)

    if metrics_df is not None:
        if not metrics_df.empty:
            print("\n--- Calculated Metrics per Tenant ID ---")
            print(metrics_df.to_string(index=False))

            # Save the results to a new CSV file
            output_filename = 'metrics_summary.csv'
            metrics_df.to_csv(output_filename, index=False)
            print(f"\nResults have been saved to '{output_filename}'")
        else:
            print("\nAnalysis complete. No data was processed or no tenant IDs were found.")