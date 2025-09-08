#!/usr/bin/env python3
"""
Fully Automated and Interactive Prediction & Evaluation Script (Single Tenant)

1. Prompts the user for fileTypeId, tenantId, and integrationId.
2. Automatically finds and loads 'tenantInformation' from local files.
3. Creates a unique output folder for the run: outputs/<fileTypeId>/<tenantId>/
4. Fetches prediction data and saves it to the unique folder.
5. Compares predictions against a ground_truth.json located in the same unique folder.
6. Saves the final report to the unique folder.
"""

import json
import csv
import os
import glob
import requests
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


def find_tenant_info(tenant_id_to_find: str, tenant_info_folder: str) -> Optional[Dict[str, Any]]:
    """Searches all JSON files in a directory to find a tenant's information by their ID."""
    print(f"\n🔍 Searching for Tenant Info for '{tenant_id_to_find}' in '{tenant_info_folder}'...")
    json_files = glob.glob(os.path.join(tenant_info_folder, "*.json"))
    if not json_files:
        print(f"   ✗ No JSON files found in the specified tenant info folder.")
        return None
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if tenant_id_to_find in data:
                print(f"   ✓ Found tenant information in {os.path.basename(file_path)}.")
                return data[tenant_id_to_find]
        except (json.JSONDecodeError, IOError) as e:
            print(f"   ⚠️ Warning: Could not read or parse {file_path}. Error: {e}")
    print(f"   ✗ Tenant Info for '{tenant_id_to_find}' not found in any files.")
    return None

def fetch_and_save_predictions(api_endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], num_iterations: int, output_dir: str) -> List[str]:
    print(f"\n🚀 Starting to fetch {num_iterations} predictions from API...")
    saved_files = []
    if not os.path.exists(output_dir):
        print(f"   Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    for i in range(num_iterations):
        iteration_num = i + 1
        print(f"   Fetching iteration {iteration_num}/{num_iterations}...")
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            prediction_data = response.json()
            file_path = os.path.join(output_dir, f"iter{iteration_num}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, indent=4)
            print(f"   ✓ Successfully saved to {file_path}")
            saved_files.append(file_path)
        except requests.exceptions.HTTPError as http_err:
            print(f"   ✗ HTTP error: {http_err}"); print(f"   Response body: {response.text}"); return []
        except requests.exceptions.RequestException as req_err:
            print(f"   ✗ Request failed: {req_err}"); return []
        except json.JSONDecodeError:
            print(f"   ✗ Failed to decode JSON. Response text: {response.text}"); return []
    print(f"✅ Finished fetching all predictions.")
    return saved_files

def generate_exhaustive_field_list(instances_json_path: str) -> List[str]:
    print(f"\n🔍 Generating exhaustive field list from: {instances_json_path}")
    try:
        with open(instances_json_path, "r", encoding='utf-8') as f: data = json.load(f)
        unique_keys = set()
        for integration in data:
            for key in integration.keys():
                if key != "fileTransferFields": unique_keys.add(key)
            fields = integration.get("fileTransferFields", [])
            for field in fields:
                key = field.get("key")
                if key: unique_keys.add(key)
        unique_keys_list = sorted(list(unique_keys))
        print(f"✓ Found {len(unique_keys_list)} unique fields.")
        return unique_keys_list
    except FileNotFoundError: print(f"✗ FATAL ERROR: Instances file not found at: {instances_json_path}"); return []
    except Exception as e: print(f"✗ FATAL ERROR: Could not process instances file. Error: {e}"); return []

class PredictionComparator:
    def __init__(self, gt_json_path: str, prediction_paths: List[str], all_possible_fields: List[str]):
        self.gt_path = gt_json_path
        self.pr_paths = prediction_paths
        self.exhaustive_fields = set(field.lower() for field in all_possible_fields)
        self.llm_model = self.setup_llm()
        self.ignored_fields = {
            "filedestination", "filename", "lookback", "integrationmode",
            "notificationemailaddress", "notificationstart", "notificationwarning",
            "notificationfailure", "notificationsuccess", "notificationcreateticket",
            "requiresfileinput", "eventinginfo", "eventbasedtriggers",
            "publishingconfiguration", "integrationid", "tenantid",
            "integrationname", "vendor", "filetypeid", "customername",
            "createddate", "updateddate", "deleted", "runtype",
            "runautomatically", "cron", "_source_file", "_tenant_id", "ui_statusmap", "globaltenantid"
        }
    def setup_llm(self) -> Optional[Any]:
        if not LLM_AVAILABLE: return None
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key: print("⚠️ Warning: GEMINI_API_KEY environment variable not set. LLM features disabled."); return None
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            print("✓ LLM (Gemini) configured successfully.")
            return model
        except Exception as e: print(f"✗ Error configuring LLM: {e}"); return None
    def _load_prediction_data(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            flattened_data = {}
            for key, value in data.items():
                if key.lower() != 'suggestions': flattened_data[key.lower()] = value
            if 'suggestions' in data and isinstance(data.get('suggestions'), dict):
                for key, value in data['suggestions'].items(): flattened_data[key.lower()] = value
            return flattened_data
        except Exception as e: print(f"✗ Error loading prediction file {file_path}: {e}"); return {}
    def _load_ground_truth_data(self) -> Dict[str, Any]:
        gt_data = {}
        try:
            with open(self.gt_path, 'r', encoding='utf-8') as f: instance = json.load(f)
            for key, value in instance.items():
                if key.lower() not in ['filetransferfields', 'deleted'] and value is not None: gt_data[key.lower()] = value
            if 'fileTransferFields' in instance and isinstance(instance['fileTransferFields'], list):
                for field in instance['fileTransferFields']:
                    field_name, field_value = field.get('key'), field.get('value')
                    if field_name: gt_data[field_name.lower()] = field_value
            return {k.lower(): v for k, v in gt_data.items()}
        except Exception as e: print(f"✗ Error loading ground truth file {self.gt_path}: {e}"); return {}
    
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
                status, match_type = "", "N"
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
                                    match_type = "GT Present PR Present and match"
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
        except Exception as e: print(f"✗ Error writing to CSV file {output_csv_path}: {e}")

    def _determine_match_status(self, field_name: str, gt_value: Any, pr_value: Any) -> tuple[str, str]:
        gt_str, pr_str = str(gt_value).strip(), str(pr_value).strip()
        if gt_str == pr_str: return "GT Present PR Present and match", "exact_match"
        
        if field_name.startswith('toggle-') and self._is_json_string(gt_str) and self._is_json_string(pr_str):
            try:
                gt_json, pr_json = json.loads(gt_str), json.loads(pr_str)
                if 'hidden' in gt_json and 'hidden' in pr_json:
                    if gt_json['hidden'] == pr_json['hidden']: return "GT Present PR Present and match", "toggle_match_hidden_only"
                    else: return "GT Present PR Present but mismatch", "incorrect_toggle_hidden_mismatch"
            except json.JSONDecodeError: pass
        
        if self._is_json_string(gt_str) and self._is_json_string(pr_str):
            if self._compare_json_values(gt_str, pr_str): return "GT Present PR Present and match", "json_partial_match"
        
        if self.llm_model:
            field_data = {'field_name': field_name, 'predicted_value': pr_str, 'gt_value': gt_str}
            llm_result = self._call_llm_for_match_analysis(field_data)
            if llm_result in ['default_match', 'json_partial_correct']: return "GT Present PR Present and match", llm_result
            return "GT Present PR Present but mismatch", llm_result
        return "GT Present PR Present but mismatch", "incorrect"
        
    def _call_llm_for_match_analysis(self, field_data: Dict[str, Any]) -> str:
        """
        Calls the LLM to analyze a mismatch and categorize it.
        This now contains the full, untruncated prompt.
        """
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
        1.  **no_prediction**: Return this if the "Predicted Value" is empty, null, or effectively blank.

        2.  **default_match**: Return this if the "Predicted Value" appears to be a generic default, a standard placeholder, or a common fallback value that is often used when specific data is not available.
            - Examples: "default", "standard", "weDoNotReport", "noMinimumAgeRequired", "false", "true", "N/A".
            - Use this category if the prediction is one of these common defaults, even if the GT is a more specific (but different) value.

        3.  **json_partial_correct**: Return this ONLY if both the GT and Predicted values are valid JSON strings. This category applies if the core information, key fields, or essential values within the JSON objects align, even if the overall structure, formatting, or non-essential keys differ. For example, if they represent the same filter logic but are structured differently.

        4.  **genuine_prediction**: Return this if the "Predicted Value" is a plausible, specific, and non-default attempt to match the GT, but is ultimately incorrect. This indicates the model tried to generate a real answer but failed.
            - Example GT: "ACME_CORP_PLAN_A"
            - Example Prediction: "ACME_PLAN_A" (close, but not an exact match)

        5.  **incorrect**: Return this if the prediction is clearly wrong, nonsensical for the given field, or does not fit any of the other categories.

        **Final Instruction:**
        Based on your analysis of the provided field information, return the single most appropriate category name from the list above.
        """
        try:
            response = self.llm_model.generate_content(prompt)
            return getattr(response, 'text', '').strip().lower()
        except Exception as e: print(f"  - LLM call failed for field '{field_data['field_name']}': {e}"); return "llm_error"

    def _format_value(self, value: Any) -> str:
        if value is None: return ""
        if isinstance(value, (dict, list)): return json.dumps(value)
        return str(value)
    
    def _is_json_string(self, value: str) -> bool:
        if not isinstance(value, str) or not value.startswith(('[', '{')): return False
        try: json.loads(value); return True
        except (json.JSONDecodeError, TypeError): return False
    
    
    def _compare_json_values(self, gt_json_str: str, pred_json_str: str) -> bool:
        try:
            gt_json, pred_json = json.loads(gt_json_str), json.loads(pred_json_str)
            if isinstance(gt_json, dict) and isinstance(pred_json, dict): return gt_json == pred_json
            if isinstance(gt_json, list) and isinstance(pred_json, list):
                gt_set = set(frozenset(d.items()) for d in gt_json)
                pred_set = set(frozenset(d.items()) for d in pred_json)
                return gt_set == pred_set
            return False
        except Exception: return False


def main():
    # ========================== DEFAULTS & CONFIGURATION ==========================
    DEFAULT_API_ENDPOINT = "https://tip-agent-config-service-dev.us-east4.dev.gcp.int/api/v1/metadata-prediction/predict"
    DEFAULT_NUM_ITERATIONS = 3
    DEFAULT_FILE_TYPE_ID = "usg.cigna.834-facets"
    TENANT_INFO_FOLDER = "D:\\RagaAI\\TIP\\pipelining_script\\Tenet_info"
    BASE_OUTPUT_FOLDER = "D:\\RagaAI\\TIP\\pipelining_script"
    BASE_INSTANCES_FOLDER = "D:\\RagaAI\\TIP\\pipelining_script\\priority_integration_data"
    
    # ========================== SCRIPT EXECUTION (INTERACTIVE) ==========================
    print("--- Automated Prediction & Evaluation Workflow (Single Tenant) ---")
    
    # --- Collect User Input ---
    file_type_id = input(f"Enter fileTypeId [{DEFAULT_FILE_TYPE_ID}]: ") or DEFAULT_FILE_TYPE_ID
    integration_id = "228ce4e5-a9d7-4ee6-9a75-ecdbf8eab45b"
    tenant_id = input("Enter the tenantId to process: ")
    bearer_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjdDdDZzYkx2QVYya1B1MVRicWVuSiJ9.eyJpc3MiOiJodHRwczovL3dlbGNvbWUtc3RhZ2luZy51a2cuZGV2LyIsInN1YiI6IjJhY2RjNzNmLTczYTQtNGI2Yy05OGI5LTQ5ZDZmOTQ0YWI4ZkBjbGllbnRzIiwiYXVkIjoiaHR0cHM6Ly90aXAudWtnLm5ldCIsImlhdCI6MTc1NjM2OTQzNCwiZXhwIjoxNzU2MzcxMjM0LCJzY29wZSI6IndyaXRlOnRpcCBhZG1pbjp0aXAgZXhlY3V0ZTp0aXAgcmVhZDp0aXAiLCJndHkiOiJjbGllbnQtY3JlZGVudGlhbHMiLCJhenAiOiIyYWNkYzczZi03M2E0LTRiNmMtOThiOS00OWQ2Zjk0NGFiOGYifQ.ZVLVkyQJ6kBZ8Dljgv7sz8I3dY_wXICaIkCNtSzQjr3_8I-j0cDzUSFByMsLzM7_Jmae-9qX6cUh7mBEdIuQeAbeEBD6aOttHd_8xjGD8X1FDMyxzJmlT2j8McCdswnV_3WwL0SP5p_pu-kq3jFq2UDMfQRdOZbM878Ag9AsCxVBMog_Fut5gvh7CvaLoFby29Jt2r1avYpgoYb0e_BgTBuM-uslbPIGx94IL3LptuPpeySyEqWqvcHArZqvKWY65zLBq5K2NwahtP9K3Nm19Kc1RUP5FA8Bc-2WfotmL1PWhZoMmiBQOEQdB_EezhDWeZFhVvsMPbSkI5sigeVYuA"
    
    num_iterations_str = input(f"Enter number of iterations [{DEFAULT_NUM_ITERATIONS}]: ") or str(DEFAULT_NUM_ITERATIONS)
    num_iterations = int(num_iterations_str)

    if not all([file_type_id, integration_id, tenant_id, bearer_token]):
        print("\n✗ One or more required inputs were left blank. Aborting.")
        return

    # --- Step 1: Define Dynamic File Paths based on fileTypeId AND tenantId ---
    run_output_path = os.path.join(BASE_OUTPUT_FOLDER, file_type_id, tenant_id)
    ground_truth_file = os.path.join(run_output_path, "ground_truth.json")
    output_report_file = os.path.join(run_output_path, f"coverage_report_{tenant_id}.csv")
    instances_json_path = os.path.join(BASE_INSTANCES_FOLDER, file_type_id, "instances.json")
    
    if not os.path.exists(ground_truth_file):
        print(f"\n✗ FATAL ERROR: Ground truth file not found for this tenant.")
        print(f"  Please make sure this file exists: {ground_truth_file}")
        return

    # --- Step 2: Look up Tenant and Field Information ---
    tenant_info = find_tenant_info(tenant_id, TENANT_INFO_FOLDER)
    if not tenant_info:
        print("\n✗ Could not find tenant information. Aborting workflow.")
        return

    exhaustive_field_list = generate_exhaustive_field_list(instances_json_path)
    if not exhaustive_field_list:
        print("\n✗ Could not generate the exhaustive field list. Aborting workflow.")
        return

    # --- Step 3: Construct API Request ---
    payload = {
        "globalTenantId": "TO-DO",
        "fileTypeId": file_type_id,
        "integrationId": integration_id,
        "tenantInformation": tenant_info
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    # --- Step 4: Fetch Predictions from API ---
    prediction_file_paths = fetch_and_save_predictions(
        api_endpoint=DEFAULT_API_ENDPOINT,
        headers=headers,
        payload=payload,
        num_iterations=num_iterations,
        output_dir=run_output_path
    )
    if not prediction_file_paths:
        print("\n✗ Aborting comparison because API fetching failed.")
        return
        
    # --- Step 5: Run Comparison ---
    comparator = PredictionComparator(
        gt_json_path=ground_truth_file,
        prediction_paths=prediction_file_paths,
        all_possible_fields=exhaustive_field_list
    )
    comparator.compare_and_generate_report(output_report_file)
    print("\n🎉 Workflow complete!")

if __name__ == "__main__":
    main()