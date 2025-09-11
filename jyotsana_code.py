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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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
                               time_log_file: str) -> tuple[Optional[str], float]:
    """Fetches predictions from API and saves them to a file. Returns (file_path, latency_seconds)."""
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
        return file_path, duration
    except requests.exceptions.HTTPError as http_err:
        print(f"   ✗ HTTP error: {http_err}");
        print(f"   Response body: {response.text}");
        return None, 0.0
    except requests.exceptions.RequestException as req_err:
        print(f"   ✗ Request failed: {req_err}");
        return None, 0.0
    except json.JSONDecodeError:
        print(f"   ✗ Failed to decode JSON. Response text: {response.text}");
        return None, 0.0


def log_prediction_time(file_path: str, file_type_id: str, tenant_id: str, duration: float):
    """Logs prediction time to a CSV file."""
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'FileTypeID', 'TenantID', 'PredictionTimeSeconds'])
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), file_type_id, tenant_id, f"{duration:.4f}"])


def calculate_metrics_from_csv(csv_file_path: str) -> Dict[str, Any]:
    """
    Calculates accuracy, coverage, and extra fields metrics from the coverage report CSV.

    Args:
        csv_file_path: Path to the coverage report CSV file

    Returns:
        Dictionary containing calculated metrics
    """
    print(f"\n📊 Calculating metrics from: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)

        # Get the status column (assuming it's Status_v1 for single iteration)
        status_col = 'Status_v1' if 'Status_v1' in df.columns else 'Status'

        # Calculate counts for different status types
        gt_present_pr_present_match = len(df[df[status_col] == "GT Present PR Present and match"])
        gt_present_pr_present_mismatch = len(df[df[status_col] == "GT Present PR Present but mismatch"])
        gt_present_pr_absent = len(df[df[status_col] == "GT Present PR Absent"])
        gt_absent_pr_present = len(df[df[status_col] == "GT Absent PR Present"])
        gt_absent_pr_absent = len(df[df[status_col] == "GT Absent PR Absent"])

        # Calculate Coverage
        coverage_numerator = gt_present_pr_present_match + gt_present_pr_present_mismatch
        coverage_denominator = coverage_numerator + gt_present_pr_absent

        if coverage_denominator > 0:
            coverage = coverage_numerator / coverage_denominator
        else:
            coverage = 0.0

        # Calculate Accuracy
        accuracy_numerator = gt_present_pr_present_match
        accuracy_denominator = gt_present_pr_present_match + gt_present_pr_present_mismatch

        if accuracy_denominator > 0:
            accuracy = accuracy_numerator / accuracy_denominator
        else:
            accuracy = 0.0

        # Extra fields predicted (GT Absent PR Present)
        extra_fields_count = gt_absent_pr_present
        extra_fields_list = df[df[status_col] == "GT Absent PR Present"]['FieldName'].tolist()

        # Total fields analyzed
        total_fields = len(df)

        metrics = {
            'total_fields': total_fields,
            'gt_present_pr_present_match': gt_present_pr_present_match,
            'gt_present_pr_present_mismatch': gt_present_pr_present_mismatch,
            'gt_present_pr_absent': gt_present_pr_absent,
            'gt_absent_pr_present': gt_absent_pr_present,
            'gt_absent_pr_absent': gt_absent_pr_absent,
            'coverage': round(coverage, 4),
            'accuracy': round(accuracy, 4),
            'extra_fields_count': extra_fields_count,
            'extra_fields_list': extra_fields_list
        }

        print(f"   ✓ Coverage: {coverage:.4f} ({coverage_numerator}/{coverage_denominator})")
        print(f"   ✓ Accuracy: {accuracy:.4f} ({accuracy_numerator}/{accuracy_denominator})")
        print(f"   ✓ Extra fields predicted: {extra_fields_count}")

        return metrics

    except Exception as e:
        print(f"   ✗ Error calculating metrics: {e}")
        return {
            'total_fields': 0,
            'gt_present_pr_present_match': 0,
            'gt_present_pr_present_mismatch': 0,
            'gt_present_pr_absent': 0,
            'gt_absent_pr_present': 0,
            'gt_absent_pr_absent': 0,
            'coverage': 0.0,
            'accuracy': 0.0,
            'extra_fields_count': 0,
            'extra_fields_list': []
        }


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
# --- PredictionComparator Class (Evaluation Logic) - FIXED ---
# ==============================================================================
class PredictionComparator:
    """Handles comparison between ground truth and predictions."""

    def __init__(self, gt_json_path: str, exhaustive_fields: List[str]):
        self.gt_path = gt_json_path
        self.exhaustive_fields = set(field.lower() for field in exhaustive_fields)
        self.llm_model = self._setup_llm()
        self.ignored_fields = IGNORED_FIELDS

    def _setup_llm(self) -> Optional[Any]:
        if not LLM_AVAILABLE:
            print("⚠️ Warning: 'google-generativeai' package not found. LLM features disabled.")
            return None
        try:
            # Try multiple ways to get the API key
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                # Try to read from a .env file or config file
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    api_key = os.environ.get("GEMINI_API_KEY")
                except:
                    pass

            if not api_key:
                print("⚠️ Warning: GEMINI_API_KEY not found in environment variables.")
                print("   To enable LLM features, set the GEMINI_API_KEY environment variable or")
                print("   add it to a .env file in the script directory.")
                return None

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            print("✓ LLM (Gemini) configured successfully.")
            return model
        except Exception as e:
            print(f"✗ Error configuring LLM: {e}")
            return None

    def _load_prediction_data(self, file_path: str) -> Dict[str, Any]:
        """Load prediction data from a specific file, ensuring clean state."""
        try:
            print(f"   Loading prediction data from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Create a fresh dictionary for this prediction file
            flattened_data = {}

            # Process top-level fields (excluding suggestions)
            for key, value in data.items():
                if key.lower() != 'suggestions':
                    flattened_data[key.lower()] = value

            # Process suggestions if they exist
            if 'suggestions' in data and isinstance(data.get('suggestions'), dict):
                for key, value in data['suggestions'].items():
                    flattened_data[key.lower()] = value

            print(f"   ✓ Loaded {len(flattened_data)} fields from prediction file")
            return flattened_data

        except Exception as e:
            print(f"✗ Error loading prediction file {file_path}: {e}")
            return {}

    def _load_ground_truth_data(self) -> Dict[str, Any]:
        """Load ground truth data, ensuring clean state."""
        try:
            print(f"   Loading ground truth data from: {self.gt_path}")
            with open(self.gt_path, 'r', encoding='utf-8') as f:
                instance = json.load(f)

            # Create a fresh dictionary for ground truth
            gt_data = {}

            # Process top-level fields
            for key, value in instance.items():
                if key.lower() not in ['filetransferfields', 'deleted'] and value is not None:
                    gt_data[key.lower()] = value

            # Process fileTransferFields
            if 'fileTransferFields' in instance and isinstance(instance['fileTransferFields'], list):
                for field in instance['fileTransferFields']:
                    field_name = field.get('key')
                    field_value = field.get('value')
                    if field_name:
                        gt_data[field_name.lower()] = field_value

            print(f"   ✓ Loaded {len(gt_data)} fields from ground truth file")
            return gt_data

        except Exception as e:
            print(f"✗ Error loading ground truth file {self.gt_path}: {e}")
            return {}

    def compare_single_prediction(self, prediction_file_path: str, output_csv_path: str):
        """Compare a single prediction file against ground truth."""
        print(f"\n🔄 Starting comparison for: {prediction_file_path}")

        # Load data fresh for each comparison
        gt_data = self._load_ground_truth_data()
        pr_data = self._load_prediction_data(prediction_file_path)

        if not gt_data:
            print("✗ Aborting due to error in loading ground truth data.")
            return

        # Determine all fields to check
        all_fields_to_check = self.exhaustive_fields.union(gt_data.keys()).union(pr_data.keys())
        final_fields = sorted([f for f in all_fields_to_check if f.lower() not in self.ignored_fields])

        print(f"✓ Comparing {len(final_fields)} fields")
        print(f"   GT fields: {len(gt_data)}")
        print(f"   PR fields: {len(pr_data)}")

        report_data = []

        for field in final_fields:
            gt_present = field in gt_data
            gt_value = gt_data.get(field)

            pr_present = field in pr_data
            pr_value = pr_data.get(field)
            if pr_value=="none":
                status = "GT Absent PR Absent"
                match_type = "N/A"
            # Determine status
            if gt_present and pr_present:
                status, match_type = self._determine_match_status(field, gt_value, pr_value)
            elif gt_present and not pr_present:
                status = "GT Present PR Absent"
                match_type = "N/A"
                # Special handling for toggle fields
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
                match_type = "N/A"
            else:  # not gt_present and not pr_present
                status = "GT Absent PR Absent"
                match_type = "N/A"

            row = {
                "FieldName": field,
                "Ground_Truth_Value": self._format_value(gt_value),
                "Predicted_Value_v1": self._format_value(pr_value),
                "Status_v1": status,
                "Match_Type_v1": match_type
            }
            report_data.append(row)

        # Write the report
        if not report_data:
            print("⚠️ No data to write to report.")
            return

        try:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=report_data[0].keys())
                writer.writeheader()
                writer.writerows(report_data)
            print(f"\n✅ Successfully generated comparison report: {output_csv_path}")
        except Exception as e:
            print(f"✗ Error writing to CSV file {output_csv_path}: {e}")

    def _determine_match_status(self, field_name: str, gt_value: Any, pr_value: Any) -> tuple[str, str]:
        gt_str = str(gt_value).strip()
        pr_str = str(pr_value).strip()

        # Exact match
        if gt_str == pr_str:
            return "GT Present PR Present and match", "exact_match"

        # Toggle field special handling
        if field_name.startswith('toggle-') and self._is_json_string(gt_str) and self._is_json_string(pr_str):
            try:
                gt_json = json.loads(gt_str)
                pr_json = json.loads(pr_str)
                if 'hidden' in gt_json and 'hidden' in pr_json:
                    if gt_json['hidden'] == pr_json['hidden']:
                        return "GT Present PR Present and match", "toggle_match_hidden_only"
                    else:
                        return "GT Present PR Present but mismatch", "incorrect_toggle_hidden_mismatch"
            except json.JSONDecodeError:
                pass

        # JSON comparison
        if self._is_json_string(gt_str) and self._is_json_string(pr_str):
            if self._compare_json_values(gt_str, pr_str):
                return "GT Present PR Present and match", "json_partial_match"

        # LLM analysis
        if self.llm_model:
            field_data = {
                'field_name': field_name,
                'predicted_value': pr_str,
                'gt_value': gt_str
            }
            llm_result = self._call_llm_for_match_analysis(field_data)
            if llm_result in ['json_partial_correct']:
                return "GT Present PR Present and match", llm_result
            return "GT Present PR Present but mismatch", llm_result

        return "GT Present PR Present but mismatch", "incorrect"

    def _call_llm_for_match_analysis(self, field_data: Dict[str, Any]) -> str:
        if not self.llm_model:
            return "incorrect"

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
        - json_partial_correct
        - incorrect
        **Analysis Rules & Categorization:**
        1. **no_prediction**: Return this if the "Predicted Value" is empty, null, or effectively blank.
        2. **json_partial_correct**: Return this ONLY if both the GT and Predicted values have matching 'values' or 'tables'...
        3. **incorrect**: Return this if the prediction is clearly wrong...
        **Final Instruction:**
        Based on your analysis... return the single most appropriate category name...
        """

        try:
            response = self.llm_model.generate_content(prompt)
            return getattr(response, 'text', '').strip().lower()
        except Exception as e:
            print(f"  - LLM call failed for field '{field_data['field_name']}': {e}")
            return "llm_error"

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def _is_json_string(self, value: str) -> bool:
        if not isinstance(value, str) or not value.startswith(('[', '{')):
            return False
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def _compare_json_values(self, gt_json_str: str, pred_json_str: str) -> bool:
        try:
            gt_json = json.loads(gt_json_str)
            pred_json = json.loads(pred_json_str)

            if isinstance(gt_json, dict) and isinstance(pred_json, dict):
                return gt_json == pred_json

            if isinstance(gt_json, list) and isinstance(pred_json, list):
                gt_set = set(frozenset(d.items()) if isinstance(d, dict) else d for d in gt_json)
                pred_set = set(frozenset(d.items()) if isinstance(d, dict) else d for d in pred_json)
                return gt_set == pred_set

            return False
        except Exception:
            return False


# ==============================================================================
# --- MAIN ORCHESTRATION FUNCTION - UPDATED ---
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
        if f.lower().endswith(('.xlsx', '.csv', '.txt', 'docx', 'pdf')) and f.lower() != 'instances.json'
    ])

    if not discovered_files:
        print("✗ No account structure files (.xlsx, .csv, .txt, .docx, .pdf) found in the directory. Aborting.")
        return

    print("   ✓ Found the following account files to process (in order):")
    for idx, filename in enumerate(discovered_files):
        print(f"     {idx + 1}. {filename}")

    # --- Step 2: Get all Integration IDs at once ---
    print("\nPlease provide the corresponding integrationId for each file listed above.")
    integration_ids_input = input("Enter a comma-separated list of integrationIds: ")
    integration_ids = [i.strip().strip('[](){}') for i in integration_ids_input.split(',')]

    if len(integration_ids) != len(discovered_files):
        print(
            f"\n✗ Error: You provided {len(integration_ids)} integrationIds, but {len(discovered_files)} files were found.")
        print("   The number of IDs must match the number of files. Aborting.")
        return

    accounts_to_process = list(zip(discovered_files, integration_ids))

    bearer_token = input("Enter your Authorization Bearer Token for the prediction API: ")

    # Check for Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        gemini_key = input("Enter your Gemini API Key (or press Enter to skip LLM features): ").strip()
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            print("✓ Gemini API key set for this session.")
        else:
            print("⚠️ LLM features will be disabled for this run.")

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
    all_tenants_metrics_data = []
    all_latency_data = []

    # --- Step 4: Main Loop to Process Each Discovered Account ---
    for i, (account_filename, integration_id) in enumerate(accounts_to_process):
        print("\n" + "=" * 70)
        print(f"Processing Account {i + 1}/{len(accounts_to_process)}: {account_filename}")
        print("=" * 70)

        tenant_id = get_tenant_id_from_csv(ACCOUNTS_CSV_PATH, file_type_id, account_filename)
        if not tenant_id:
            print(f"✗ Skipping this account: tenantId could not be found.")
            continue

        run_output_path = os.path.join(BASE_OUTPUT_FOLDER, file_type_id, tenant_id)
        ground_truth_file = os.path.join(run_output_path, "ground_truth.json")
        if not create_ground_truth_from_instances(instances_json_path, tenant_id, ground_truth_file):
            print(f"✗ Skipping this account: could not create its ground truth file.")
            continue

        source_file_to_upload = os.path.join(accounts_path, account_filename)
        file_extension = os.path.splitext(account_filename)[1]
        destination_filename = f"{UPLOAD_FILENAME_PREFIX_UUID}_{file_type_id}_{integration_id}{file_extension}"
        if not upload_account_structure_file(
                UPLOAD_API_ENDPOINT, source_file_to_upload, destination_filename,
                GCS_BUCKET_NAME, "account_structure"):
            print(f"✗ Skipping this account due to file upload failure.")
            continue

        # Use hardcoded tenant info as in original script
        tenant_info = {
            "tenantName": "ARN1027 suitesupport02 UAT",
            "tenantAlias": "arn1027_suitesupport02_uat",
            "tenantCreatedDateTime": "2025-05-14T15:37:16Z",
            "tenantModifiedDateTime": "2025-05-14T15:37:16Z",
            "region": "US",
            "tenantType": "UAT",
            "globalTenantId": "ca28b853-27c3-433f-93eb-541f835269a6",
            "customerId": "be4d8751-3105-43ed-b9a1-e57c9b31b9d2",
            "customerName": "suitesupport02",
            "status": "Active",
            "verticalMarket": "Unknown",
            "segment": "Enterprise",
            "products": ["PRO"],
            "dataCenter": "US-EAST4",
            "clientAccessKey": "30UIK",
            "currencyIsoCode": "Unknown",
            "language": "Unknown",
            "employeeCount": 0,
            "salesforceId": "",
            "customerCreatedDateTime": "2025-05-07T14:31:49Z",
            "customerModifiedDateTime": "2025-05-07T14:31:49Z",
            "hxUrl": "https://g02i00extueslb.dev.us.corp"
        }

        payload = {
            "globalTenantId": "TO-DO",
            "fileTypeId": file_type_id,
            "integrationId": integration_id,
            "tenantInformation": tenant_info
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {bearer_token}"}

        prediction_file_path, prediction_latency = fetch_and_save_predictions(
            PREDICTION_API_ENDPOINT, headers, payload, run_output_path, TIME_LOG_FILE
        )
        if not prediction_file_path:
            print(f"✗ Skipping evaluation due to API fetching failure.")
            continue

        # Store main prediction latency data
        all_latency_data.append({
            'tenantId': tenant_id,
            'accountStructureFile': account_filename,
            'fileTypeId': file_type_id,
            'integrationId': integration_id,
            'api_endpoint': 'main',
            'latency_seconds': prediction_latency
        })

        if run_local_prediction == 'yes':
            print("\n- Running prediction against local server for logging...")
            local_prediction_file, local_latency = fetch_and_save_predictions(LOCAL_API_ENDPOINT, headers, payload,
                                                                              run_output_path, TIME_LOG_FILE)
            if local_prediction_file:
                all_latency_data.append({
                    'tenantId': tenant_id,
                    'accountStructureFile': account_filename,
                    'fileTypeId': file_type_id,
                    'integrationId': integration_id,
                    'api_endpoint': 'local',
                    'latency_seconds': local_latency
                })

        # --- FIXED: Create a new comparator instance for each account ---
        output_report_file = os.path.join(run_output_path, f"coverage_report_{tenant_id}.csv")

        # Create fresh comparator instance for this specific account
        comparator = PredictionComparator(
            gt_json_path=ground_truth_file,
            exhaustive_fields=exhaustive_field_list
        )

        # Use the new single prediction comparison method
        comparator.compare_single_prediction(prediction_file_path, output_report_file)

        if os.path.exists(output_report_file):
            df = pd.read_csv(output_report_file)
            df.insert(0, 'tenantId', tenant_id)
            df.insert(1, 'accountStructureFile', account_filename)
            all_tenants_report_data.append(df)

            # Calculate metrics for this tenant
            metrics = calculate_metrics_from_csv(output_report_file)
            metrics['tenantId'] = tenant_id
            metrics['accountStructureFile'] = account_filename
            metrics['fileTypeId'] = file_type_id
            metrics['integrationId'] = integration_id
            metrics['prediction_latency_seconds'] = prediction_latency
            all_tenants_metrics_data.append(metrics)

    # --- Step 5: Create Consolidated Report ---
    if all_tenants_report_data:
        print("\n" + "=" * 70)
        print("Creating Consolidated Report...")
        print("=" * 70)
        consolidated_df = pd.concat(all_tenants_report_data, ignore_index=True)
        consolidated_report_path = os.path.join(BASE_OUTPUT_FOLDER, file_type_id, "consolidated_report.csv")
        consolidated_df.to_csv(consolidated_report_path, index=False)
        print(f"✅ Consolidated report for {len(all_tenants_report_data)} tenants saved to: {consolidated_report_path}")

    # --- Step 6: Create Metrics Summary Report ---
    if all_tenants_metrics_data:
        print("\n" + "=" * 70)
        print("Creating Metrics Summary Report...")
        print("=" * 70)

        # Create a summary DataFrame with key metrics
        metrics_summary = []
        for metrics in all_tenants_metrics_data:
            summary_row = {
                'tenantId': metrics['tenantId'],
                'accountStructureFile': metrics['accountStructureFile'],
                'fileTypeId': metrics['fileTypeId'],
                'integrationId': metrics['integrationId'],
                'total_fields': metrics['total_fields'],
                'coverage': metrics['coverage'],
                'accuracy': metrics['accuracy'],
                'extra_fields_count': metrics['extra_fields_count'],
                'prediction_latency_seconds': metrics.get('prediction_latency_seconds', 0.0),
                'gt_present_pr_present_match': metrics['gt_present_pr_present_match'],
                'gt_present_pr_present_mismatch': metrics['gt_present_pr_present_mismatch'],
                'gt_present_pr_absent': metrics['gt_present_pr_absent'],
                'gt_absent_pr_present': metrics['gt_absent_pr_present'],
                'gt_absent_pr_absent': metrics['gt_absent_pr_absent'],
                'extra_fields_list': '; '.join(metrics['extra_fields_list']) if metrics['extra_fields_list'] else ''
            }
            metrics_summary.append(summary_row)

        metrics_df = pd.DataFrame(metrics_summary)
        metrics_report_path = os.path.join(BASE_OUTPUT_FOLDER, file_type_id, "metrics_summary.csv")
        metrics_df.to_csv(metrics_report_path, index=False)
        print(f"✅ Metrics summary for {len(all_tenants_metrics_data)} tenants saved to: {metrics_report_path}")

        # Create separate latency report
        if all_latency_data:
            latency_df = pd.DataFrame(all_latency_data)
            latency_report_path = os.path.join(BASE_OUTPUT_FOLDER, file_type_id, "latency_report.csv")
            latency_df.to_csv(latency_report_path, index=False)
            print(f"✅ Latency report for {len(all_latency_data)} API calls saved to: {latency_report_path}")

        # Print summary statistics
        if len(metrics_summary) > 0:
            avg_coverage = metrics_df['coverage'].mean()
            avg_accuracy = metrics_df['accuracy'].mean()
            avg_latency = metrics_df['prediction_latency_seconds'].mean()
            total_extra_fields = metrics_df['extra_fields_count'].sum()
            print(f"\n📊 Summary Statistics:")
            print(f"   Average Coverage: {avg_coverage:.4f}")
            print(f"   Average Accuracy: {avg_accuracy:.4f}")
            print(f"   Average Prediction Latency: {avg_latency:.4f} seconds")
            print(f"   Total Extra Fields Predicted: {total_extra_fields}")

    print("\n" + "=" * 70)
    print("🎉 Full Pipelining Workflow Complete!")
    print("=" * 70)


# ==============================================================================
# --- SCRIPT ENTRY POINT ---
# ==============================================================================

if __name__ == "__main__":
    main()