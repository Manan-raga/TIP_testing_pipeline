#!/usr/bin/env python3
"""
Fully Automated End-to-End Pipelining and Evaluation Script (Advanced)

Orchestrates the entire testing workflow:
1.  Automatically discovers account structure files for a given fileTypeId.
2.  Prompts for a single list of corresponding integrationIds.
3.  Looks up tenantIds from a central CSV file.
4.  For each account:
    a. Automatically generates the ground_truth.json.
    b. Uploads the account structure file with the required nomenclature.
    c. Fetches prediction data from the API and logs the request time.
    d. Saves results in a unique folder: outputs/<fileTypeId>/<tenantId>/
    e. Runs a full comparison and evaluation, creating an individual report.
5.  Finally, creates a single consolidated CSV report for all tenants tested.
"""

import json
import csv
import os
import glob
import requests
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# --- Library Setup ---
try:
    import google.generativeai as genai

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ Warning: 'google-generativeai' package not found. LLM features disabled.")

load_dotenv()

# ==============================================================================
# --- CONFIGURATION CONSTANTS ---
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTION_API_ENDPOINT = "https://tip-agent-config-service-dev.us-east4.dev.gcp.int/api/v1/metadata-prediction/predict"
UPLOAD_API_ENDPOINT = "https://tip-config-prediction-service-dev.us-east4.dev.gcp.int/upload"
LOCAL_API_ENDPOINT = "http://localhost:8080/api/v1/metadata-prediction/predict"
DEFAULT_FILE_TYPE_ID = "usg.lincoln.supp-life"
GCS_BUCKET_NAME = "tip_input_data"
UPLOAD_FILENAME_PREFIX_UUID = "ca28b853-27c3-433f-93eb-541f835269a6"

ACCOUNTS_CSV_PATH = os.path.join(SCRIPT_DIR, "accounts_to_run.csv")
TENANT_INFO_FOLDER = os.path.join(SCRIPT_DIR, "Tenet_info")
BASE_OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "outputs")
BASE_INSTANCES_FOLDER = os.path.join(SCRIPT_DIR, "priority_integration_data")
TIME_LOG_FILE = os.path.join(SCRIPT_DIR, "prediction_times.csv")

IGNORED_FIELDS = {
    "filedestination", "filename", "lookback", "integrationmode",
    "notificationemailaddress", "notificationstart", "notificationwarning",
    "notificationfailure", "notificationsuccess", "notificationcreateticket",
    "requiresfileinput", "eventinginfo", "eventbasedtriggers",
    "publishingconfiguration", "integrationid", "tenantid",
    "integrationname", "vendor", "filetypeid", "customername",
    "createddate", "updateddate", "deleted", "runtype",
    "runautomatically", "cron", "_source_file", "_tenant_id",
    "ui_statusmap", "globaltenantid"
}


# ==============================================================================
# --- PIPELINE HELPER FUNCTIONS ---
# ==============================================================================

def create_ground_truth_from_instances(instances_json_path: str, tenant_id: str, output_path: str) -> bool:
    """
    Finds a tenant's data in instances.json and saves it as ground_truth.json.
    """
    print(f"\n📄 Generating ground_truth.json for tenant '{tenant_id}'...")
    try:
        with open(instances_json_path, "r", encoding='utf-8') as f:
            all_instances = json.load(f)

        tenant_instance_data = None
        for instance in all_instances:
            if instance.get("tenantId") == tenant_id:
                tenant_instance_data = instance
                break

        if not tenant_instance_data:
            print(f"   ✗ Tenant '{tenant_id}' not found in '{instances_json_path}'.")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tenant_instance_data, f, indent=4)

        print(f"   ✓ Successfully created ground truth file at: {output_path}")
        return True

    except FileNotFoundError:
        print(f"   ✗ FATAL ERROR: The instances file was not found at '{instances_json_path}'.")
        return False
    except Exception as e:
        print(f"   ✗ FATAL ERROR: Failed to create ground truth file. Error: {e}")
        return False


def get_tenant_id_from_csv(csv_path: str, file_type_id: str, account_name: str) -> Optional[str]:
    """
    Reads the accounts_to_run.csv to find the tenantId for a given account,
    ignoring the file extension of the account_name.
    """
    print(f"\n🔍 Searching for account name (without extension) in '{csv_path}'...")
    account_name_without_extension = os.path.splitext(account_name)[0]
    print(f"   Looking for File Type: '{file_type_id}', Account Name: '{account_name_without_extension}'")

    try:
        df = pd.read_csv(csv_path)
        result = df[
            (df['fileTypeId'] == file_type_id) &
            (df['account_structure_name'] == account_name_without_extension)
            ]
        if not result.empty:
            tenant_id = result.iloc[0]['tenantId']
            print(f"   ✓ Found tenantId: {tenant_id}")
            return str(tenant_id)
        else:
            print(f"   ✗ No entry found for the given file type and account name.")
            return None
    except FileNotFoundError:
        print(f"   ✗ FATAL ERROR: The account mapping file was not found at '{csv_path}'.")
        return None
    except Exception as e:
        print(f"   ✗ FATAL ERROR: Failed to read or process the CSV file. Error: {e}")
        return None


def upload_account_structure_file(
        upload_endpoint: str, source_file_path: str, destination_filename: str,
        bucket_name: str, blob_key_prefix: str
) -> bool:
    """Uploads a file using a multipart/form-data request."""
    print(f"\n☁️ Uploading account structure file...")
    print(f"   Source: {source_file_path}")
    print(f"   Uploading As: {destination_filename}")
    if not os.path.exists(source_file_path):
        print(f"   ✗ File not found at the source path. Cannot upload.")
        return False

    files = {'file': (destination_filename, open(source_file_path, 'rb'),
                      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    data = {'bucket_name': bucket_name, 'blob_key_prefix': blob_key_prefix}
    try:
        response = requests.post(upload_endpoint, files=files, data=data, verify=False)
        response.raise_for_status()
        print(f"   ✓ File uploaded successfully (Status: {response.status_code}).")
        print(f"   Server Response: {response.text}")
        return True
    except requests.exceptions.HTTPError as http_err:
        print(f"   ✗ HTTP error during file upload: {http_err}");
        print(f"   Response body: {response.text}");
        return False
    except requests.exceptions.RequestException as req_err:
        print(f"   ✗ Request failed during file upload: {req_err}");
        return False
    finally:
        if 'file' in files:
            files['file'][1].close()


def find_tenant_info(tenant_id_to_find: str, tenant_info_folder: str) -> Optional[Dict[str, Any]]:
    """Searches for tenant information in JSON files within the specified folder."""
    print(f"\n🔍 Searching for Tenant Info for '{tenant_id_to_find}' in '{tenant_info_folder}'...")
    json_files = glob.glob(os.path.join(tenant_info_folder, "*.json"))
    if not json_files:
        print(f"   ✗ No JSON files found in the specified tenant info folder.")
        return None
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if tenant_id_to_find in data:
                print(f"   ✓ Found tenant information in {os.path.basename(file_path)}.")
                return data[tenant_id_to_find]
        except (json.JSONDecodeError, IOError) as e:
            print(f"   ⚠️ Warning: Could not read or parse {file_path}. Error: {e}")
    print(f"   ✗ Tenant Info for '{tenant_id_to_find}' not found in any files.")
    return None


def fetch_and_save_predictions(api_endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], output_dir: str,
                               time_log_file: str) -> Optional[str]:
    """Fetches predictions from API and saves them to a file."""
    print(f"\n🚀 Starting to fetch prediction from API: {api_endpoint}")
    if not os.path.exists(output_dir):
        print(f"   Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    start_time = time.time()
    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, verify=False)
        duration = time.time() - start_time
        print(f"   API call took: {duration:.4f} seconds.")
        log_prediction_time(time_log_file, payload.get('fileTypeId'),
                            payload.get('tenantInformation', {}).get('globalTenantId'), duration)
        response.raise_for_status()
        prediction_data = response.json()
        file_path = os.path.join(output_dir, "iter1.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, indent=4)
        print(f"   ✓ Successfully saved prediction to {file_path}")
        return file_path
    except requests.exceptions.HTTPError as http_err:
        print(f"   ✗ HTTP error: {http_err}");
        print(f"   Response body: {response.text}");
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"   ✗ Request failed: {req_err}");
        return None
    except json.JSONDecodeError:
        print(f"   ✗ Failed to decode JSON. Response text: {response.text}");
        return None


def log_prediction_time(file_path: str, file_type_id: str, tenant_id: str, duration: float):
    """Logs prediction time to a CSV file."""
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'FileTypeID', 'TenantID', 'PredictionTimeSeconds'])
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), file_type_id, tenant_id, f"{duration:.4f}"])


def generate_exhaustive_field_list(instances_json_path: str) -> List[str]:
    """Generates a comprehensive list of all unique fields from instances.json."""
    print(f"\n🔍 Generating exhaustive field list from: {instances_json_path}")
    try:
        with open(instances_json_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        unique_keys = set()
        for integration in data:
            for key in integration.keys():
                if key != "fileTransferFields":
                    unique_keys.add(key)
            fields = integration.get("fileTransferFields", [])
            for field in fields:
                key = field.get("key")
                if key:
                    unique_keys.add(key)
        unique_keys_list = sorted(list(unique_keys))
        print(f"✓ Found {len(unique_keys_list)} unique fields.")
        return unique_keys_list
    except FileNotFoundError:
        print(f"✗ FATAL ERROR: Instances file not found at: {instances_json_path}")
        return []
    except Exception as e:
        print(f"✗ FATAL ERROR: Could not process instances file. Error: {e}")
        return []


# ==============================================================================
# --- PredictionComparator Class (Evaluation Logic) ---
# ==============================================================================
class PredictionComparator:
    """Handles comparison between ground truth and predictions."""

    def __init__(self, gt_json_path: str, prediction_paths: List[str], all_possible_fields: List[str]):
        self.gt_path = gt_json_path
        self.pr_paths = prediction_paths
        self.exhaustive_fields = set(field.lower() for field in all_possible_fields)
        self.llm_model = self._setup_llm()
        self.ignored_fields = IGNORED_FIELDS

    def _setup_llm(self) -> Optional[Any]:
        if not LLM_AVAILABLE: return None
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key: print(
                "⚠️ Warning: GEMINI_API_KEY environment variable not set. LLM features disabled."); return None
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            print("✓ LLM (Gemini) configured successfully.")
            return model
        except Exception as e:
            print(f"✗ Error configuring LLM: {e}"); return None

    def _load_prediction_data(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            flattened_data = {}
            for key, value in data.items():
                if key.lower() != 'suggestions': flattened_data[key.lower()] = value
            if 'suggestions' in data and isinstance(data.get('suggestions'), dict):
                for key, value in data['suggestions'].items(): flattened_data[key.lower()] = value
            return flattened_data
        except Exception as e:
            print(f"✗ Error loading prediction file {file_path}: {e}"); return {}

    def _load_ground_truth_data(self) -> Dict[str, Any]:
        gt_data = {}
        try:
            with open(self.gt_path, 'r', encoding='utf-8') as f:
                instance = json.load(f)
            for key, value in instance.items():
                if key.lower() not in ['filetransferfields', 'deleted'] and value is not None: gt_data[
                    key.lower()] = value
            if 'fileTransferFields' in instance and isinstance(instance['fileTransferFields'], list):
                for field in instance['fileTransferFields']:
                    field_name, field_value = field.get('key'), field.get('value')
                    if field_name: gt_data[field_name.lower()] = field_value
            return {k.lower(): v for k, v in gt_data.items()}
        except Exception as e:
            print(f"✗ Error loading ground truth file {self.gt_path}: {e}"); return {}

    def compare_and_generate_report(self, output_csv_path: str):
        print("\n🔄 Starting comparison process...")
        gt_data = self._load_ground_truth_data()
        pr_data_list = [self._load_prediction_data(path) for path in self.pr_paths]
        if not gt_data: print("✗ Aborting due to error in loading ground truth data."); return
        all_fields_to_check = self.exhaustive_fields.union(gt_data.keys())
        for pr_data in pr_data_list: all_fields_to_check.update(pr_data.keys())
        final_fields = sorted([f for f in all_fields_to_check if f.lower() not in self.ignored_fields])
        print(f"✓ Comparing a total of {len(final_fields)} fields across {len(self.pr_paths)} iterations.")
        report_data = []
        for field in final_fields:
            gt_present, gt_value = field in gt_data, gt_data.get(field)
            row = {"FieldName": field, "Ground_Truth_Value": self._format_value(gt_value)}
            for i, pr_data in enumerate(pr_data_list):
                version = i + 1
                pr_present, pr_value = field in pr_data, pr_data.get(field)
                status, match_type = "", "N/A"
                if gt_present and pr_present:
                    status, match_type = self._determine_match_status(field, gt_value, pr_value)
                elif gt_present and not pr_present:
                    status = "GT Present PR Absent"
                    if field.startswith('toggle-'):
                        gt_str = self._format_value(gt_value)
                        if self._is_json_string(gt_str):
                            try:
                                gt_json = json.loads(gt_str)
                                if gt_json.get('hidden') is True:
                                    match_type = "correctly_absent_as_hidden"
                            except json.JSONDecodeError:
                                pass
                elif not gt_present and pr_present:
                    status = "GT Absent PR Present"
                elif not gt_present and not pr_present:
                    status = "GT Absent PR Absent"
                row[f"Predicted_Value_v{version}"] = self._format_value(pr_value)
                row[f"Status_v{version}"] = status
                row[f"Match_Type_v{version}"] = match_type
            report_data.append(row)
        if not report_data: print("⚠️ No data to write to report."); return
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=report_data[0].keys())
                writer.writeheader()
                writer.writerows(report_data)
            print(f"\n✅ Successfully generated comparison report: {output_csv_path}")
        except Exception as e:
            print(f"✗ Error writing to CSV file {output_csv_path}: {e}")

    def _determine_match_status(self, field_name: str, gt_value: Any, pr_value: Any) -> tuple[str, str]:
        gt_str, pr_str = str(gt_value).strip(), str(pr_value).strip()
        if gt_str == pr_str: return "GT Present PR Present and match", "exact_match"
        if field_name.startswith('toggle-') and self._is_json_string(gt_str) and self._is_json_string(pr_str):
            try:
                gt_json, pr_json = json.loads(gt_str), json.loads(pr_str)
                if 'hidden' in gt_json and 'hidden' in pr_json:
                    if gt_json['hidden'] == pr_json['hidden']:
                        return "GT Present PR Present and match", "toggle_match_hidden_only"
                    else:
                        return "GT Present PR Present but mismatch", "incorrect_toggle_hidden_mismatch"
            except json.JSONDecodeError:
                pass
        if self._is_json_string(gt_str) and self._is_json_string(pr_str):
            if self._compare_json_values(gt_str, pr_str): return "GT Present PR Present and match", "json_partial_match"
        if self.llm_model:
            field_data = {'field_name': field_name, 'predicted_value': pr_str, 'gt_value': gt_str}
            llm_result = self._call_llm_for_match_analysis(field_data)
            if llm_result in ['default_match',
                              'json_partial_correct']: return "GT Present PR Present and match", llm_result
            return "GT Present PR Present but mismatch", llm_result
        return "GT Present PR Present but mismatch", "incorrect"

    def _call_llm_for_match_analysis(self, field_data: Dict[str, Any]) -> str:
        if not self.llm_model: return "incorrect"
        prompt = f"""
        Analyze the predicted value compared to the ground truth for a given field and categorize the prediction.
        **Field Information:**
        - Field Name: "{field_data['field_name']}"
        - Ground Truth (GT) Value: "{field_data['gt_value']}"
        - Predicted Value: "{field_data['predicted_value']}"
        **Your Task:**
        Evaluate the "Predicted Value" against the "Ground Truth (GT) Value" based on the rules below.
        You must return ONLY ONE of the following category names as your response:
        - no_prediction
        - default_match
        - json_partial_correct
        - genuine_prediction
        - incorrect
        **Analysis Rules & Categorization:**
        1. **no_prediction**: Return this if the "Predicted Value" is empty, null, or effectively blank.
        2. **default_match**: Return this if the "Predicted Value" appears to be a generic default...
        3. **json_partial_correct**: Return this ONLY if both the GT and Predicted values are valid JSON strings...
        4. **genuine_prediction**: Return this if the "Predicted Value" is a plausible, specific, and non-default attempt...
        5. **incorrect**: Return this if the prediction is clearly wrong...
        **Final Instruction:**
        Based on your analysis... return the single most appropriate category name...
        """
        try:
            response = self.llm_model.generate_content(prompt)
            return getattr(response, 'text', '').strip().lower()
        except Exception as e:
            print(f"  - LLM call failed for field '{field_data['field_name']}': {e}"); return "llm_error"

    def _format_value(self, value: Any) -> str:
        if value is None: return ""
        if isinstance(value, (dict, list)): return json.dumps(value)
        return str(value)

    def _is_json_string(self, value: str) -> bool:
        if not isinstance(value, str) or not value.startswith(('[', '{')): return False
        try:
            json.loads(value); return True
        except (json.JSONDecodeError, TypeError):
            return False

    def _compare_json_values(self, gt_json_str: str, pred_json_str: str) -> bool:
        try:
            gt_json, pred_json = json.loads(gt_json_str), json.loads(pred_json_str)
            if isinstance(gt_json, dict) and isinstance(pred_json, dict): return gt_json == pred_json
            if isinstance(gt_json, list) and isinstance(pred_json, list):
                gt_set = set(frozenset(d.items()) if isinstance(d, dict) else d for d in gt_json)
                pred_set = set(frozenset(d.items()) if isinstance(d, dict) else d for d in pred_json)
                return gt_set == pred_set
            return False
        except Exception:
            return False


# ==============================================================================
# --- MAIN ORCHESTRATION FUNCTION ---
# ==============================================================================

def main():
    """Main orchestration function for the pipeline."""
    print("=" * 70)
    print("--- Automated Pipelining & Evaluation Workflow (Advanced) ---")
    print("=" * 70)

    # --- Step 1: Collect Common Inputs & Discover Accounts ---
    file_type_id = input(f"Enter the fileTypeId to process [{DEFAULT_FILE_TYPE_ID}]: ") or DEFAULT_FILE_TYPE_ID

    accounts_path = os.path.join(BASE_INSTANCES_FOLDER, file_type_id)
    print(f"\n📂 Discovering account structure files in: {accounts_path}")
    if not os.path.isdir(accounts_path):
        print(f"✗ Directory not found. Aborting.")
        return

    discovered_files = sorted([
        f for f in os.listdir(accounts_path)
        if f.lower().endswith(('.xlsx', '.csv', '.txt')) and f.lower() != 'instances.json'
    ])

    if not discovered_files:
        print("✗ No account structure files (.xlsx, .csv, .txt) found in the directory. Aborting.")
        return

    print("   ✓ Found the following account files to process (in order):")
    for idx, filename in enumerate(discovered_files):
        print(f"     {idx + 1}. {filename}")

    # --- Step 2: Get all Integration IDs at once ---
    print("\nPlease provide the corresponding integrationId for each file listed above.")
    integration_ids_input = input("Enter a comma-separated list of integrationIds: ")
    integration_ids = [i.strip() for i in integration_ids_input.split(',')]

    if len(integration_ids) != len(discovered_files):
        print(
            f"\n✗ Error: You provided {len(integration_ids)} integrationIds, but {len(discovered_files)} files were found.")
        print("   The number of IDs must match the number of files. Aborting.")
        return

    accounts_to_process = list(zip(discovered_files, integration_ids))

    bearer_token = input("Enter your Authorization Bearer Token for the prediction API: ")
    run_local_prediction = input("Run prediction against local server for logs? (yes/no) [no]: ").lower() or "no"
    if run_local_prediction == 'yes':
        print("\n⚠️ Please ensure local services are running with the latest code.")
        input("   Press Enter to continue...")

    # --- Step 3: Setup for the Run ---
    instances_json_path = os.path.join(BASE_INSTANCES_FOLDER, file_type_id, "instances.json")
    exhaustive_field_list = generate_exhaustive_field_list(instances_json_path)
    if not exhaustive_field_list:
        print("\n✗ Aborting: Could not generate the field list from instances.json.")
        return

    all_tenants_report_data = []

    # --- Step 4: Main Loop to Process Each Discovered Account ---
    for i, (account_filename, integration_id) in enumerate(accounts_to_process):
        print("\n" + "=" * 70)
        print(f"Processing Account {i + 1}/{len(accounts_to_process)}: {account_filename}")
        print("=" * 70)

        tenant_id = get_tenant_id_from_csv(ACCOUNTS_CSV_PATH, file_type_id, account_filename)
        if not tenant_id:
            print(f"✗ Skipping this account: tenantId could not be found.");
            continue

        run_output_path = os.path.join(BASE_OUTPUT_FOLDER, file_type_id, tenant_id)
        ground_truth_file = os.path.join(run_output_path, "ground_truth.json")
        if not create_ground_truth_from_instances(instances_json_path, tenant_id, ground_truth_file):
            print(f"✗ Skipping this account: could not create its ground truth file.");
            continue

        source_file_to_upload = os.path.join(accounts_path, account_filename)
        file_extension = os.path.splitext(account_filename)[1]
        destination_filename = f"{UPLOAD_FILENAME_PREFIX_UUID}_{file_type_id}_{integration_id}{file_extension}"
        if not upload_account_structure_file(
                UPLOAD_API_ENDPOINT, source_file_to_upload, destination_filename,
                GCS_BUCKET_NAME, "account_structure"):
            print(f"✗ Skipping this account due to file upload failure.");
            continue

        tenant_info = find_tenant_info(tenant_id, TENANT_INFO_FOLDER)
        if not tenant_info:
            print(f"✗ Skipping this account: tenant info could not be found.");
            continue

        payload = {"globalTenantId": "TO-DO", "fileTypeId": file_type_id, "integrationId": integration_id,
                   "tenantInformation": tenant_info}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {bearer_token}"}
        prediction_file_path = fetch_and_save_predictions(
            PREDICTION_API_ENDPOINT, headers, payload, run_output_path, TIME_LOG_FILE
        )
        if not prediction_file_path:
            print(f"✗ Skipping evaluation due to API fetching failure.");
            continue

        if run_local_prediction == 'yes':
            print("\n- Running prediction against local server for logging...")
            fetch_and_save_predictions(LOCAL_API_ENDPOINT, headers, payload, run_output_path, TIME_LOG_FILE)

        output_report_file = os.path.join(run_output_path, f"coverage_report_{tenant_id}.csv")
        comparator = PredictionComparator(
            gt_json_path=ground_truth_file,
            prediction_paths=[prediction_file_path],
            all_possible_fields=exhaustive_field_list
        )
        comparator.compare_and_generate_report(output_report_file)

        if os.path.exists(output_report_file):
            df = pd.read_csv(output_report_file)
            df.insert(0, 'tenantId', tenant_id)
            df.insert(1, 'accountStructureFile', account_filename)
            all_tenants_report_data.append(df)

    # --- Step 5: Create Consolidated Report ---
    if all_tenants_report_data:
        print("\n" + "=" * 70)
        print("Creating Consolidated Report...")
        print("=" * 70)
        consolidated_df = pd.concat(all_tenants_report_data, ignore_index=True)
        consolidated_report_path = os.path.join(BASE_OUTPUT_FOLDER, file_type_id, "consolidated_report.csv")
        consolidated_df.to_csv(consolidated_report_path, index=False)
        print(f"✅ Consolidated report for {len(all_tenants_report_data)} tenants saved to: {consolidated_report_path}")

    print("\n" + "=" * 70)
    print("🎉 Full Pipelining Workflow Complete!")
    print("=" * 70)


# ==============================================================================
# --- SCRIPT ENTRY POINT ---
# ==============================================================================

if __name__ == "__main__":
    main()